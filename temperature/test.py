#Here is we calculate smooth calibration error for original, temperature scaling and isotronic regression.
import fire
import os
import torch
import torchvision as tv
from torch.utils.data.sampler import SubsetRandomSampler
from models import DenseNet
from temperature_scaling import ModelWithTemperature
import numpy as np
import cvxpy as cp
from torch.nn import functional as F
import math
import statistics

def test_ECE(valid_loader, model, iso = None, b = False):
    x = []
    y = []
    n_bins = 15
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    self_bin_lowers = bin_boundaries[:-1]
    self_bin_uppers = bin_boundaries[1:]
    logits_list = []
    labels_list = []
    with torch.no_grad():
        for input, label in valid_loader:
            input = input.cuda()
            logits = model(input)
            logits_list.append(logits)
            labels_list.append(label)
        logits = torch.cat(logits_list).cuda()
        labels = torch.cat(labels_list).cuda()

        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)
        if b == True:
            return (confidences.cpu().numpy(), accuracies.cpu().numpy())

        if iso != None:
            confidences = iso.predict(confidences.cpu().numpy())
            ece = 0
            for bin_lower, bin_upper in zip(self_bin_lowers, self_bin_uppers):
                # Calculated |confidence - accuracy| in each bin
                in_bin = np.greater(confidences,bin_lower.item()) * np.less_equal(confidences,bin_upper.item())
                prop_in_bin = in_bin.mean()
                if prop_in_bin.item() > 0:
                    accuracy_in_bin = accuracies[in_bin].float().mean()
                    avg_confidence_in_bin = confidences[in_bin].mean()
                    x.append(avg_confidence_in_bin)
                    y.append(accuracy_in_bin.cpu().numpy())
                    ece += (avg_confidence_in_bin - accuracy_in_bin.cpu().numpy()) * prop_in_bin
            print('ece for iso regression')
            print(ece)
            return (np.array(x), np.array(y)), ece
            

        for bin_lower, bin_upper in zip(self_bin_lowers, self_bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                x.append(avg_confidence_in_bin.cpu().numpy())
                y.append(accuracy_in_bin.cpu().numpy())
    return (np.array(x), np.array(y))

def smCE_LP(S):
    (x_list, y_list) = S
    indices = np.argsort(x_list)
    x_list = x_list[indices]
    y_list = y_list[indices]
    A = np.diag([1]*len(x_list))
    A = (A -np.roll(A, 1, axis = 1))[0:len(x_list)-1]
    A = np.concatenate([A, -A])
    b = (x_list- np.roll(x_list,1,axis=0))[1:len(x_list)]
    b = np.concatenate([b,b])
    c = y_list - x_list
    n = len(x_list)
    x = cp.Variable(n)
    objective = cp.Minimize(np.array([c/n]) @ x)
    constraints = [-1 <= x, x <= 1, A@x <= b]
    prob = cp.Problem(objective, constraints)
    result = prob.solve()
    return -result

def test_calibration(loader, model):
    model.eval()
    print('Evaluating')

    total_error = 0
    counter=0
    x_list = np.zeros(256*19)
    y_list = np.zeros(256*19)
    for i, (input, target) in enumerate(loader):
        with torch.no_grad():
            # Forward pass
            input = input.cuda()
            target = target.cuda()
            output = model(input)
            
        # Accounting
        _, predictions = torch.topk(output, 1)
        target = target.cpu().numpy()/100
        predictions = predictions.cpu().numpy().reshape(len(target))/100
        if i <=18:
            x_list[i*256:(i+1)*256] = predictions
            y_list[i*256:(i+1)*256] = target
        #cal_error = smCE_LP((predictions, target))
        #total_error += cal_error

        # Log errors
        #print('Calibration_error '+str(i) +' '+ str(cal_error))
        counter+=1
    
    cal_error = smCE_LP((x_list, y_list))
    return cal_error
    #return total_error/counter


def demo(data, save, depth=40, growth_rate=12, batch_size=256):
    """
    Applies temperature scaling to a trained model.

    Takes a pretrained DenseNet-CIFAR100 model, and a validation set
    (parameterized by indices on train set).
    Applies temperature scaling, and saves a temperature scaled version.

    NB: the "save" parameter references a DIRECTORY, not a file.
    In that directory, there should be two files:
    - model.pth (model state dict)
    - valid_indices.pth (a list of indices corresponding to the validation set).

    data (str) - path to directory where data should be loaded from/downloaded
    save (str) - directory with necessary files (see above)
    """
    # Load model state dict
    model_filename = os.path.join(save, 'model.pth')
    if not os.path.exists(model_filename):
        raise RuntimeError('Cannot find file %s to load' % model_filename)
    state_dict = torch.load(model_filename)

    # Load validation indices
    valid_indices_filename = os.path.join(save, 'valid_indices.pth')
    #valid_indices_filename = os.path.join(save, 'model_with_temperature.pth')
    if not os.path.exists(valid_indices_filename):
        raise RuntimeError('Cannot find file %s to load' % valid_indices_filename)
    valid_indices = torch.load(valid_indices_filename)

    # Regenerate validation set loader
    mean = [0.5071, 0.4867, 0.4408]
    stdv = [0.2675, 0.2565, 0.2761]
    test_transforms = tv.transforms.Compose([
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(mean=mean, std=stdv),
    ])
    train_transforms = tv.transforms.Compose([
        tv.transforms.RandomCrop(32, padding=4),
        tv.transforms.RandomHorizontalFlip(),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(mean=mean, std=stdv),
    ])
    train_set = tv.datasets.CIFAR100(data, train=True, transform=train_transforms, download=True)
    train_loader = torch.utils.data.DataLoader(train_set, pin_memory=True, batch_size=batch_size)


    # Load original model
    if (depth - 4) % 3:
        raise Exception('Invalid depth')
    block_config = [(depth - 4) // 6 for _ in range(3)]
    orig_model = DenseNet(
        growth_rate=growth_rate,
        block_config=block_config,
        num_classes=100
    ).cuda()
    orig_model.load_state_dict(state_dict)
    #valid_results = test_calibration(valid_loader, orig_model)
    from sklearn.isotonic import IsotonicRegression
    orig_result = []
    iso_result = []
    temp_result = []
    total_ece = 0
    for _ in range(20):
        valid_set = tv.datasets.CIFAR100(data, train=True, transform=test_transforms, download=True)
        valid_loader = torch.utils.data.DataLoader(valid_set, pin_memory=True, batch_size=batch_size,
                                                   sampler=SubsetRandomSampler(valid_indices))
        train_set = tv.datasets.CIFAR100(data, train=True, transform=train_transforms, download=True)
        train_loader = torch.utils.data.DataLoader(train_set, pin_memory=True, batch_size=batch_size)
        cal = test_ECE(train_loader, orig_model, b = True)
        #valid_results = smCE_LP(cal)
        #orig_result.append(valid_results)
        ir = IsotonicRegression().fit(cal[0], cal[1])
        
        valid_set_2 = tv.datasets.CIFAR100(data, train=True, transform=test_transforms, download=True)
        valid_loader_2 = torch.utils.data.DataLoader(valid_set_2, pin_memory=True, batch_size=batch_size,
                                                   sampler=SubsetRandomSampler(valid_indices))
        S_calibrated, ece = test_ECE(valid_loader_2, orig_model, ir)
        total_ece+=ece
        #f, y = (ir.predict(S_calibrated[0]),S_calibrated[1])
        #for i in range(f.shape[0]):
        #    if math.isnan(f[i]):
        #        f[i] = 0
        #    else:
        #        break
        #for i in range(f.shape[0]-1,0,-1):
        #    if math.isnan(f[i]):
        #        f[i] = 1
        #    else:
        #        break


        #S_calibrated =(f,y)
        iso_result.append(smCE_LP(S_calibrated))
        # Now we're going to wrap the model with a decorator that adds temperature scaling
        model = ModelWithTemperature(orig_model)
        # Tune the model temperature, and save the results
        #valid_set_3 = tv.datasets.CIFAR100(data, train=True, transform=test_transforms, download=True)
        #valid_loader_3 = torch.utils.data.DataLoader(valid_set_3, pin_memory=True, batch_size=batch_size,
        #                                       sampler=SubsetRandomSampler(valid_indices))
        train_set_2 = tv.datasets.CIFAR100(data, train=True, transform=train_transforms, download=True)
        train_loader_2 = torch.utils.data.DataLoader(train_set_2, pin_memory=True, batch_size=batch_size)
        model.set_temperature(train_loader_2)
        valid_set_1 = tv.datasets.CIFAR100(data, train=True, transform=test_transforms, download=True)
        valid_loader_1 = torch.utils.data.DataLoader(valid_set_1, pin_memory=True, batch_size=batch_size,
                                               sampler=SubsetRandomSampler(valid_indices))
        cal = test_ECE(valid_loader_1, model)
        valid_results = smCE_LP(cal)
        temp_result.append(valid_results)

    #print('orgin')
    #print(orig_result)
    #print(statistics.median(orig_result))
    print('iso')
    print(iso_result)
    print(statistics.median(iso_result))
    print(total_ece/20)
    print('temp')
    print(temp_result)
    print(statistics.median(temp_result))





    #valid_results = test_calibration(valid_loader, model)
    #print('Done with temperatrue!' + str(valid_results))


if __name__ == '__main__':
    """
    Applies temperature scaling to a trained model.

    Takes a pretrained DenseNet-CIFAR100 model, and a validation set
    (parameterized by indices on train set).
    Applies temperature scaling, and saves a temperature scaled version.

    NB: the "save" parameter references a DIRECTORY, not a file.
    In that directory, there should be two files:
    - model.pth (model state dict)
    - valid_indices.pth (a list of indices corresponding to the validation set).

    --data (str) - path to directory where data should be loaded from/downloaded
    --save (str) - directory with necessary files (see above)
    """
    fire.Fire(demo)
