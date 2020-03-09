import csv
def csv_record(data,path):
    with open(path+"record.csv", "a+") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(data)

import torch
def gae(sample, last_value, gamma, lam):
    running_step = sample["s"].size()[0]
    sample["advs"] = torch.zeros((running_step, 1), dtype=torch.float32)
    sample["return"] = torch.zeros((running_step, 1), dtype=torch.float32)

    last_gaelam = 0
    last_return = 0
    value = torch.cat((sample["value"], last_value), dim=0)
    for t in reversed(range(running_step)):
        sample["return"][t] = last_return = sample["r"][t] + gamma * last_return * (1-sample["tr"][t])
        delta = sample["r"][t] + gamma * value[t+1] * (1-sample["tr"][t]) - value[t]
        last_gaelam = delta + gamma * lam * (1-sample["tr"][t]) * last_gaelam
        sample["advs"][t] = last_gaelam
    # sample["return"] = sample["advs"]+sample["value"]

    adv = sample["advs"]   # Normalize the advantages
    adv = (adv - torch.mean(adv))/(torch.std(adv)+1e-8)
    sample["advs"] = adv
    mean_ep_reward = torch.sum(sample["r"])/torch.sum(torch.eq(sample["tr"],1))
    print("the runner have sampled "+str(running_step)+" data and the mean_ep_reward is ", mean_ep_reward)
    return sample