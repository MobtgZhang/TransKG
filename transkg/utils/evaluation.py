import numpy as np
from tqdm import tqdm
def calSimilarity(expTailMatrix:np.ndarray,tailEmbedding:np.ndarray,simMeasure="dot"):
    '''
    This is the similarity of the tail vector and real one.
    :param expTailMatrix: shape of (N,emb_dim) Calculate by head and relation,
    N is the sample num(or batch_size).
    :param tailEmbedding: shape of (entityNum,emb_dim) The entity embedding matrix,
    entityNum is the number of entities.
    :param simMeasure: the method of calculate similarity.
    :return: shape of (N,entityNum) The similarity between each vector in expTailMatrix
    and all vectors in tailEmbedding.
    '''
    simMeasure = simMeasure.lower()
    assert (simMeasure.lower() in {"dot","cos","l2","l1"})
    if simMeasure == "l1":
        simScore = []
        for expM in expTailMatrix:
            score = np.linalg.norm(expM[np.newaxis, :] - tailEmbedding, ord=1, axis=1, keepdims=False)
            simScore.append(score)
        return np.array(simScore)
    elif simMeasure == "l2":
        simScore = []
        for expM in expTailMatrix:
            score = np.linalg.norm(expM[np.newaxis, :] - tailEmbedding, ord=2, axis=1, keepdims=False)
            simScore.append(score)
        return np.array(simScore)
    elif simMeasure == "dot":
        return np.matmul(expTailMatrix,tailEmbedding.T)
    else: # cos
        aScore = expTailMatrix / np.linalg.norm(expTailMatrix,ord=2,axis=1,keepdims=True)
        bScore = tailEmbedding / np.linalg.norm(expTailMatrix,ord=2,axis=1,keepdims=True)
        return np.matmul(aScore,bScore).T
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
    realScore = simScore[np.arange(tail.shape[0]),tail].reshape((-1,1))
    judMatrix = simScore - realScore
    if simMeasure in {"dot","cos"}:
        judMatrix[judMatrix>0] =1
        judMatrix[judMatrix<0] =0
        score = np.sum(judMatrix,axis=1)
        return score
    else: # L1 or L2
        judMatrix[judMatrix > 0] = 0
        judMatrix[judMatrix < 0] = 1
        score = np.sum(judMatrix, axis=1)
        return score
def calHyperSim(expTailMatrix,tailEmbedding,hyperMatrix,simMeasure="dot"):
    '''
    The method to calculate the TransH model that contains hyper parameters.
    In every step in zip(expTailMatrix,tailEmbedding) do the following step:
    Step1: Projection tailEmbedding on hyperM as hyperEmbedding(shape as (N,emb_dim))
    Step2: Calculate similarity between expTailMatrix and hyperTailEmbedding.
    Step3: Add similarity to simScore.
    (1,E) * malmul((N,E),(E,1)) --> (1,E) * (N,1) --> (N,E)
    :param expTailMatrix: The head and relation embeddings added matrix.
    :param tailEmbedding: The real tail embeddings.
    :param hyperMatrix: Hyper parameters in TransH model.
    :param simMeasure: The method to calculate the simScore.
    :return: The list of the similarity score.
    '''
    simMeasure = simMeasure.lower()
    assert (simMeasure.lower() in {"dot", "cos", "l2", "l1"})
    simScore = []
    for expM,hypM in zip(expTailMatrix,hyperMatrix):
        hyperEmbedding = tailEmbedding - hypM[np.newaxis,:] * np.matmul(tailEmbedding,hypM[:,np.newaxis])
        if simScore == "dot":
            simScore.append(np.squeeze(np.matmul(hyperEmbedding,expM[:,np.newaxis])))
        elif simScore == "l2":
            score = np.linalg.norm(hyperEmbedding-expM[np.newaxis,:],ord=2,axis=1,keepdims=False)
            simScore.append(score)
        else:
            print("ERROR : simMeasure %s is not supported!"%simMeasure)
            exit(1)
    return np.array(simScore)
def calMapSim(expTailMatrix,tailEmbedding,tailMapMatrix,relMapMatrix,simMeasure="L2"):
    '''
    This method aims to calculate the score of the TransD model.
    :param expTailMatrix:
    :param tailEmbedding:
    :param tailMapMatrix:
    :param relMapMatrix:
    :param simMeasure:
    :return:
    '''
    simMeasure = simMeasure.lower()
    assert ( simMeasure in {"dot", "cos", "l2", "l1"})
    simScore = []
    ent_dim = tailMapMatrix.shape[1]
    rel_dim = relMapMatrix.shape[1]
    for expM,relMap in zip(expTailMatrix,relMapMatrix):
        Mrt = np.matmul(relMap[np.newaxis,:,np.newaxis],tailMapMatrix[:,np.newaxis,:]) + np.eye(rel_dim,ent_dim)
        if simMeasure == "l2":
            score = np.linalg.norm(np.squeeze(np.matmul(Mrt, tailEmbedding[:, :, np.newaxis]), axis=2) - expM, ord=2,
                                   axis=1, keepdims=False)
            simScore.append(score)
        elif simMeasure == "l1":
            score = np.linalg.norm(np.squeeze(np.matmul(Mrt, tailEmbedding[:, :, np.newaxis]), axis=2) - expM, ord=1,
                                   axis=1, keepdims=False)
            simScore.append(score)
        else:
            print("ERROR : SimMeasure %s is not supported!" % simMeasure)
            exit(1)
    return np.array(simScore)
def calWeightSim(expTailMatrix,tailEmbedding,Wr):
    '''
    Weights calculating for the TransA model.
    :param expTailMatrix:
    :param tailEmbedding:
    :param Wr: The Wr weight matrix.
    :return: Return the score of the TransA model.
    '''
    simScore = []
    for expM in expTailMatrix:
        error = np.abs(tailEmbedding-expM)
        score = np.squeeze(np.matmul(np.matmul(error[:,np.newaxis,:],Wr),error[:,:,np.newaxis]))
        simScore.append(score)
    return np.array(simScore)
def calKLSim(headMatrix,headCoMatrix,relationMatrix,relationCoMatrix,tailMatrix,tailCoMatrix,simMeasure="KL"):
    '''
    :param headMatrix:
    :param headCoMatrix:
    :param relationMatrix:
    :param relationCoMatrix:
    :param tailMatrix:
    :param tailCoMatrix:
    :param simMeasure:
    :return:
    '''
    assert (simMeasure in {"KL","EL"})
    simScore = []
    for hM,hC,rM,rC in zip(headMatrix,headCoMatrix,relationMatrix,relationCoMatrix):
        errorm = tailMatrix - hM
        errorv = tailCoMatrix + hC
        if simMeasure == "KL":
            score1 = np.sum(errorv/rC,axis=1,keepdims=False) + \
                np.sum((rM-errorm)**2/rC,axis=1,keepdims=True)
            score2 = np.sum(rC/errorv,axis=1,keepdims=False) + \
                np.sum((rM-errorm)**2/errorv,axis=1,keepdims=True)
            simScore.append((score1+score2)/2)
        elif simMeasure == "EL":
            score1 = np.sum((errorm-rM)**2/(errorv+rC),axis=1,keepdims=False)
            score2 = np.sum(np.log(errorv+rC),axis=1,keepdims=False)
            simScore.append((score1+score2)/2)
        else:
            print("ERROR : SimMeasure %s is not supported!" % simMeasure)
            exit(1)
    return np.array(simScore)
def evalTransE(head,relation,tail,simMeasure,**kargs):
    '''
    This is the eval method of TransE.
    :param head: The entity of the dataset.
    :param relation: The relation of the dataset.
    :param tail: The tail of the dataset.
    :param simMeasure: The measure of the similarity,including "cos","dot","L1","L2"
    :param kargs: The additional arguments of function.
    :return: The rank score of the TransE model.
    '''
    # This method is to gather the embedding together for calculating.
    head = np.take(kargs["entEmbedding"],indices=head,axis=0)
    relation = np.take(kargs["relEmbedding"],indices=relation,axis=0)
    # tail = np.take(kargs["entityEmbed"],indices=tail,axis=0)
    # Calculate the similarity score and get the rank.
    simScore = calSimilarity(head+relation,kargs["entEmbedding"],simMeasure=simMeasure)
    return simScore
def evalTransH(head,relation,tail,simMeasure,**kargs):
    '''
    This is the eval method of TransH.
    Because the model has the Hyper parameters,
    it must calculate the score with the parameters.
    :param head: The entity of the dataset.
    :param relation: The relation of the dataset.
    :param tail: The tail of the dataset.
    :param simMeasure: The measure of the similarity,including "cos","dot","L1","L2"
    :param kargs: The additional arguments of function.
    :return: The rank score of the TransE model.
    '''
    # This method is to gather the embedding together for calculating.
    head = np.take(kargs["entEmbedding"], indices=head, axis=0)
    hyper = np.take(kargs["relationHyper"], indices=relation, axis=0)
    relation = np.take(kargs["relEmbedding"], indices=relation, axis=0)
    # tail = np.take(kargs["entEmbedding"], indices=tail, axis=0)
    # projection of the embedding
    head = head - hyper * np.sum(hyper*head,axis=1,keepdims=True)
    simScore = calHyperSim(head+relation,kargs["entEmbedding"],hyper,simMeasure)
    return simScore
def evalTransR(head,relation,tail,simMeasure,**kargs):
    '''
    This is the evaluation method of TransR model.
    :param head: The entity of the dataset.
    :param relation: The relation of the dataset.
    :param tail: The tail of the dataset.
    :param simMeasure: The measure of the similarity,including "cos","dot","L1","L2".
    :param kargs: The additional arguments of function.
    :return: The rank score of the TransR model.
    '''
    # Gather all the embedding
    head = np.take(kargs["entEmbedding"],indices=head,axis=0)
    relation = np.take(kargs["relEmbedding"],indices=relation,axis=0)
    MR = kargs["transfer"]
    head = np.matmul(head,MR)
    # tail = np.matmul(tail,MR)
    simScore = calSimilarity(head+relation,kargs["entEmbedding"],simMeasure)
    return simScore
def evalTransD(head,relation,tail,simMeasure,**kargs):
    '''
    The method of calculate TransD model score.
    :param head: The entity of the dataset.
    :param relation: The relation of the dataset.
    :param tail: The tail of the dataset.
    :param simMeasure: The measure of the similarity,including "cos","dot","L1","L2".
    :param kargs: The additional arguments of function.
    :return: The rank score of the TransR model.
    '''
    # Gather embedding for TransD model.
    head = np.take(kargs["entEmbedding"], indices=head, axis=0)
    headp = np.take(kargs["entMapEmbedding"], indices=head, axis=0)
    relation = np.take(kargs["relEmbedding"], indices=relation, axis=0)
    relationp = np.take(kargs["relMapEmbedding"], indices=relation, axis=0)
    # tail = np.take(kargs["entEmbedding"], indices=tail, axis=0)
    # tailp = np.take(kargs["entMapEmbedding"], indices=tail, axis=0)
    rel_dim = relation.shape[1]
    ent_dim = head.shape[1]
    MrH = np.matmul(relationp[:,:,np.newaxis],headp[:,np.newaxis,:])+np.eye(rel_dim,ent_dim)
    # MrT = np.matmul(relationp[:, :, np.newaxis], tailp[:, np.newaxis, :]) + np.eye(rel_dim, ent_dim)
    head = np.squeeze(np.matmul(MrH,head[:,:,np.newaxis]),axis=2)
    # tail = np.squeeze(np.matmul(MrT,tail[:,:,np.newaxis]),axis=2)
    simScore = calMapSim(head+relation,kargs["entEmbedding"],kargs["entMapEmbedding"],relationp,simMeasure=simMeasure)
    return simScore
def evalTransA(head,relation,tail,**kargs):
    '''
    This method aims to calculate the score of the TransA model.
    :param head: The head dataset.
    :param relation: The relation dataset.
    :param tail: The tail dataset.
    :param kargs: The additional parameters of the model to calculate the score.
    :return: Returns the score of TransA model.
    '''
    # Gather embedding
    head = np.take(kargs["entEmbedding"],indices=head,axis=0)
    relation = np.take(kargs["relEmbedding"],indices=relation,axis=0)
    # Calculate simScore
    simScore = calWeightSim(head+relation,kargs["entEmbedding"],kargs["relWeight"])
    return simScore
def evalKG2E(head,relation,tail,**kargs):
    '''
    The method of calculating KG2E model.
    :param head: The head dataset.
    :param relation: The relation dataset.
    :param tail: The tail dataset.
    :param kargs: The additional parameters of the KG2E.
    :return: Return the score of the similarity.
    '''
    # Gather embedding
    headv = np.take(kargs["entCovar"],indices=head,axis=0)
    headm = np.take(kargs["entEmbedding"],indices=head,axis=0)
    relationv = np.take(kargs["relCovar"],indices=relation,axis=0)
    relationm = np.take(kargs["relEmbedding"],indices=relation,axis=0)
    # Calculdate simScore
    simScore = calKLSim(headm,headv,relationm,relationv,kargs["entEmbedding"],kargs["entCovar"],simMeasure=kargs["Sim"])
    return simScore
def calHit10(simScore, tail, simMeasure="dot"):
    pass
def MREvaluation(dataloader,model_name,simMeasure="dot",**kargs):
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
    for item in tqdm(dataloader,desc="mr socre process:"):
        head,relation,tail = item[:,0],item[:,1],item[:,2]
        if model_name == "TransE":
            simScore = evalTransE(head,relation,tail,simMeasure,**kargs)
        elif model_name == "TransH":
            simScore = evalTransH(head,relation,tail,simMeasure,**kargs)
        elif model_name == "TransR":
            simScore = evalTransR(head,relation,tail,simMeasure,**kargs)
        elif model_name == "TransD":
            simScore = evalTransD(head,relation,tail,simMeasure,**kargs)
        elif model_name == "TransA":
            simScore = evalTransA(head,relation,tail,**kargs)
        elif model_name == "KG2E":
            simScore = evalKG2E(head,relation,tail,**kargs)
        else:
            print("ERROR : The %s evaluation is not supported!"%model_name)
            exit(1)
        ranks = calRank(simScore, tail, simMeasure=simMeasure)
        R += np.sum(ranks)
        N += ranks.shape[0]
    return R/N
def Hit10Evaluation(dataloader,model_name,simMeasure="dot",**kargs):
    R = 0
    N = 0
    for item in tqdm(dataloader, desc="mr socre process:"):
        head, relation, tail = item[:, 0], item[:, 1], item[:, 2]
        if model_name == "TransE":
            simScore = evalTransE(head, relation, tail, simMeasure, **kargs)
        elif model_name == "TransH":
            simScore = evalTransH(head, relation, tail, simMeasure, **kargs)
        elif model_name == "TransR":
            simScore = evalTransR(head, relation, tail, simMeasure, **kargs)
        elif model_name == "TransD":
            simScore = evalTransD(head, relation, tail, simMeasure, **kargs)
        elif model_name == "TransA":
            simScore = evalTransA(head, relation, tail, **kargs)
        elif model_name == "KG2E":
            simScore = evalKG2E(head, relation, tail, **kargs)
        else:
            print("ERROR : The %s evaluation is not supported!" % model_name)
            exit(1)
        hit10score = calHit10(simScore, tail, simMeasure=simMeasure)
        R += np.sum(hit10score)
        N += hit10score.shape[0]
    return R / N
