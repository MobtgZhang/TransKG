import numpy as np
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
def calRank(simScore:np.ndarray,tail:np.ndarray,simMeasure:str):
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
    if simScore in {"dot","cos"}:
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
    head = np.take(kargs["entityEmbed"],indices=head,axis=0)
    relation = np.take(kargs["relationEmbed"],indices=relation,axis=0)
    tail = np.take(kargs["entityEmbed"],indices=tail,axis=0)
    # Calculate the similarity score and get the rank.
    simScore = calSimilarity(head+relation,kargs["entityEmbed"],simMeasure=simMeasure)
    ranks = calRank(simScore,tail,simMeasure=simMeasure)
    return ranks
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
    head = np.take(kargs["entityEmbedding"], indices=head, axis=0)
    hyper = np.take(kargs["hyperEmbedding"], indices=relation, axis=0)
    relation = np.take(kargs["relationEmbedding"], indices=relation, axis=0)
    tail = np.take(kargs["entityEmbedding"], indices=tail, axis=0)
    # projection of the embedding
    head = head - hyper * np.sum(hyper*head,axis=1,keepdims=True)
    simScore = calHyperSim(head+relation,kargs["entityEmbedding"],hyper,simMeasure)
    ranks = calRank(simScore,tail,simMeasure)
    return ranks
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
    headp = np.take()
def evalTransD():
    pass
def evalTransA():
    pass
def evalKG2E():
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
    for item in dataloader:
        head,relation,tail = item[:,0],item[:,1],item[:,2]
        if model_name == "TransE":
            ranks = None
        elif model_name == "TransH":
            ranks = None
        elif model_name == "TransR":
            ranks = None
        elif model_name == "TransD":
            ranks = None
        elif model_name == "TransA":
            ranks = None
        elif model_name == "KG2E":
            ranks = None
        else:
            print("ERROR : The %s evaluation is not supported!"%model_name)
            exit(1)
        R += np.sum(ranks)
        N += ranks.shape[0]
    return R/N
def Hit10Evaluation(dataloader,model_name,simMeasure="dot",**kwargs):
    pass
