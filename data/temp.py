import sys

def predict(x):

    if x[5] <= 9.795053004:
        if x[1] <= 9.0:
            if x[3] <= 21.0:
                if x[0] <= 65.0:
                    return 0
                else:
                    return 0
            else:
                if x[0] <= 48.0:
                    return 1
                else:
                    return 0
        else:
            if x[3] <= 13.0:
                return 0
            else:
                if x[3] <= 24.0:
                    return 1
                else:
                    return 0
    else:
        if x[1] <= 7.0:
            if x[4] <= 18.0:
                if x[4] <= 2.0:
                    return 1
                else:
                    return 0
            else:
                if x[0] <= 68.0:
                    return 1
                else:
                    return 0
        else:
            if x[1] <= 11.0:
                if x[4] <= 11.0:
                    return 1
                else:
                    return 1
            else:
                if x[3] <= 53.0:
                    return 1
                else:
                    return 0
                

x = eval(sys.argv[1])
result = predict(x)
print(result)