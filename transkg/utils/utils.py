import os
def printArgs(args):
    '''
    This method is used for print the args in the console terminal.
    :param args: the training dataset parameters
    :return:
    '''
    print("="*20 + "Arguments" + "="*20)
    argsDict = vars(args)
    for arg, value in argsDict.items():
        print("==> {} : {}".format(arg, value))
    print("="*50)
def checkPath(path, raise_error=True):
    '''
    :param path: The path for the code to check.
    :param raise_error: Whether to raise error for the path error.
    :return:
    '''
    if not os.path.exists(path):
        if raise_error:
            print("ERROR : Path %s does not exist!" % path)
            exit(1)
        else:
            print("WARNING : Path %s does not exist!" % path)
            print("INFO : Creating path %s." % path)
            os.makedirs(path)
            print("INFO : Successfully making dir!")
    return
