import csv
def csv_record(data,path):
    with open(path+"record.csv", "a+") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(data)

import torch
def gae(sample, last_value, gamma, lam):
    running_step = len(sample["s"])
    sample["advs"] = torch.zeros((running_step), dtype=torch.float32)
    value_cal = torch.cat(sample["value"]).squeeze()

    last_gaelam = 0
    last_return = 0

    value = torch.cat((value_cal, last_value))
    for t in reversed(range(running_step)):
        # sample["return"][t] = last_return = sample["r"][t] + gamma * last_return * (1-sample["tr"][t])
        delta = sample["r"][t] + gamma * value[t+1] * (1-sample["tr"][t]) - value[t]
        last_gaelam = delta + gamma * lam * (1-sample["tr"][t]) * last_gaelam
        sample["advs"][t] = last_gaelam
    sample["return"] = sample["advs"]+value_cal

    adv = sample["advs"]   # Normalize the advantages
    adv = (adv - torch.mean(adv))/(torch.std(adv)+1e-8)
    sample["advs"] = adv
    # mean_ep_reward = torch.sum(sample["r"])/torch.sum(torch.eq(sample["tr"],1))
    # print("the runner have sampled "+str(running_step)+" data and the mean_ep_reward is ", mean_ep_reward)
    return sample

def generate_reture(sample, last_value, gamma, lam):
    running_step = sample["s"].size()[0]
    sample["advs"] = torch.zeros((running_step, 1), dtype=torch.float32)
    sample["return"] = torch.zeros((running_step, 1), dtype=torch.float32)

    last_return = 0
    for t in reversed(range(running_step)):
        sample["return"][t] = last_return = sample["r"][t] + gamma * last_return * (1 - sample["tr"][t])

    r = sample["return"]
    r = (r - torch.mean(r)) / (torch.std(r) + 1e-8)
    sample["return"] = r
    sample["advs"] = sample["return"]-sample["value"]
    mean_ep_reward = torch.sum(sample["r"]) / torch.sum(torch.eq(sample["tr"], 1))
    print("the runner have sampled " + str(running_step) + " data and the mean_ep_reward is ", mean_ep_reward)
    return sample



def get_gae(rewards, masks, values, gamma, lamda):
    rewards = torch.Tensor(rewards)
    masks = torch.Tensor(masks)
    returns = torch.zeros_like(rewards)
    advants = torch.zeros_like(rewards)

    running_returns = 0
    previous_value = 0
    running_advants = 0

    for t in reversed(range(0, len(rewards))):
        running_returns = rewards[t] + gamma * running_returns * (1-masks[t])
        running_tderror = rewards[t] + gamma * previous_value * (1-masks[t]) - \
                    values[t]
        running_advants = running_tderror + gamma * lamda * \
                          running_advants * (1-masks[t])

        returns[t] = running_returns
        previous_value = values[t]
        advants[t] = running_advants

    advants = (advants - advants.mean()) / advants.std()
    return returns, advants