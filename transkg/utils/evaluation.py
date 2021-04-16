import logging
import torch
from torch.autograd import Variable
from tqdm import tqdm
logger = logging.getLogger()
def calRank(simScore,tail,simMeasure):
    '''
    This method calculate the rank of the right tail.
    :param simScore(N,entityNum): The similarity score of the entity and the relation.
    :param tail(N,): The tail of the data,real score.
    :param simMeasure: The method of calculating distance,
    including "cos","dot","L1","L2" method.
    When the larger the score, the higher/better the ranking when using "dot" or "cos",
    When the smaller the score, the higher/better the ranking when using "L1" or "L2".
    The steps of calculating score:
    Step1: Get the score of real tails.
    Step2: Get the result of (simScore - realScore).
    Step3: Count positive/negative number of each line as the rank of the real entity.
    :return: The rank score of the matrix.
    '''
    simMeasure = simMeasure.lower()
    assert (simMeasure.lower() in {"dot", "cos", "l2", "l1"})
    realScore = simScore[torch.arange(tail.shape[0]),tail].reshape((-1,1))
    judMatrix = simScore - realScore
    if simMeasure in {"dot","cos"}:
        judMatrix[judMatrix>0] =1
        judMatrix[judMatrix<0] =0
    else: # L1 or L2
        judMatrix[judMatrix > 0] = 0
        judMatrix[judMatrix < 0] = 1
    score = torch.sum(judMatrix,dim=1)
    return score
def MREvaluation(dataloader,model,simMeasure="dot",use_gpu=True):
    '''
    This is the implementation of MR metric, that is MR represents Mean Rank Metric.
    :param dataloader: The dataset loader
    :param model_name: The model name of this evaluation.
    :param simMeasure:
    :param kargs:
    :return:
    '''
    R = 0
    N = 0
    for item in tqdm(dataloader,desc="mr socre process"):
        head,relation,tail = item[:,0],item[:,1],item[:,2]
        head,relation,tail = to_var(head,use_gpu),to_var(relation,use_gpu),to_var(tail,use_gpu)
        simScore = model.predictSimScore(head,relation,simMeasure)
        ranks = calRank(simScore, tail, simMeasure=simMeasure)
        R += torch.sum(ranks)
        N += ranks.size(0)
    return R/N
def to_var( x, use_gpu):
    if use_gpu:
        return Variable(torch.LongTensor(x).cuda())
    else:
        return Variable(torch.LongTensor(x))
