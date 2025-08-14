import numpy as np

def weight(path, dtype, name, shape, arr=None):
    f = open(path, mode='w')
    if(len(shape)==1):
        if(arr!=None):
            W = np.load(arr)
        else:
            W = np.random.randint(0,high=1000,size=[shape[0]])
            # W = np.zeros(100).astype(np.int64)
        f.write(f"{dtype} {name}[{shape[0]}] = {{")
        for i in range(shape[0]-1):
            f.write(f'{W[i]}, ')
        f.write(f'{W[shape[0]-1]}')
        f.write('};\n')
    elif(len(shape)==2):
        if(arr!=None):
            W = np.load(arr)
        else:
            W = np.random.randn(shape[0],shape[1])
        f.write(f"{dtype} {name}[{shape[0]}][{shape[1]}] = {{\n")
        for i in range(shape[0]-1):
            f.write('{')
            for j in range(shape[1]-1):
                f.write(f'{W[i][j]}, ')
            f.write(f'{W[i][shape[1]-1]}}},\n')
        f.write('{')
        for j in range(W.shape[1]-1):
            f.write(f'{W[shape[0]-1][j]}, ')
        f.write(f'{W[shape[0]-1][shape[1]-1]}}}\n')
        f.write('};\n')
    elif(len(shape)==3):
        if(arr!=None):
            W = np.load(arr)
        else:
            W = np.random.randn(shape[0],shape[1],shape[2])
        f.write(f"{dtype} {name}[{shape[0]}][{shape[1]}][{shape[2]}] = {{\n")
        for i in range(shape[0]-1):
            f.write('{')
            for j in range(shape[1]-1):
                f.write('{')
                for k in range(shape[2]-1):   
                    f.write(f'{W[i][j][k]}, ')
                f.write(f'{W[i][j][shape[2]-1]}}},')
            f.write('{')
            for k in range(shape[2]-1):   
                f.write(f'{W[i][shape[1]-1][k]}, ')
            f.write(f'{W[i][shape[1]-1][shape[2]-1]}}}}},\n')
        f.write('{')
        for j in range(shape[1]-1):
            f.write('{')
            for k in range(shape[2]-1):   
                f.write(f'{W[shape[0]-1][j][k]}, ')
            f.write(f'{W[shape[0]-1][j][shape[2]-1]}}},')
        f.write('{')
        for k in range(shape[2]-1):   
            f.write(f'{W[shape[0]-1][shape[1]-1][k]}, ')
        f.write(f'{W[shape[0]-1][shape[1]-1][shape[2]-1]}}}}}\n')
        f.write('};\n')
    elif(len(shape)==4):
        if(arr!=None):
            W = np.load(arr)
        else:
            W = np.random.randn(shape[0],shape[1],shape[2],shape[3])
        f.write(f"{dtype} {name}[{shape[0]}][{shape[1]}][{shape[2]}][{shape[3]}] = {{\n")
        for i in range(shape[0]-1):
            f.write('{')
            for j in range(shape[1]-1):
                f.write('{')
                for k in range(shape[2]-1):
                    f.write('{')
                    for l in range(shape[3]-1):
                        f.write(f'{W[i][j][k][l]}, ')
                    f.write(f'{W[i][j][k][shape[3]-1]}}},')
                f.write('{')
                for l in range(shape[3]-1):   
                    f.write(f'{W[i][j][shape[2]-1][l]}, ')
                f.write(f'{W[i][j][shape[2]-1][shape[3]-1]}}}}},\n')
            f.write('{')
            for k in range(shape[2]-1):
                f.write('{')
                for l in range(shape[3]-1):
                    f.write(f'{W[i][shape[1]-1][k][l]}, ')
                f.write(f'{W[i][shape[1]-1][k][shape[3]-1]}}},')
            f.write('{')
            for l in range(shape[3]-1):   
                f.write(f'{W[i][shape[1]-1][shape[2]-1][l]}, ')
            f.write(f'{W[i][shape[1]-1][shape[2]-1][shape[3]-1]}}}}}}},\n')
        f.write('{')
        for j in range(shape[1]-1):
            f.write('{')
            for k in range(shape[2]-1):
                f.write('{')
                for l in range(shape[3]-1):
                    f.write(f'{W[shape[0]-1][j][k][l]}, ')
                f.write(f'{W[shape[0]-1][j][k][shape[3]-1]}}},')
            f.write('{')
            for l in range(shape[3]-1):   
                f.write(f'{W[shape[0]-1][j][shape[2]-1][l]}, ')
            f.write(f'{W[shape[0]-1][j][shape[2]-1][shape[3]-1]}}}}},\n')
        f.write('{')
        for k in range(shape[2]-1):
            f.write('{')
            for l in range(shape[3]-1):
                f.write(f'{W[shape[0]-1][shape[1]-1][k][l]}, ')
            f.write(f'{W[shape[0]-1][shape[1]-1][k][shape[3]-1]}}},')
        f.write('{')
        for l in range(shape[3]-1):   
            f.write(f'{W[shape[0]-1][shape[1]-1][shape[2]-1][l]}, ')
        f.write(f'{W[shape[0]-1][shape[1]-1][shape[2]-1][shape[3]-1]}}}}}}}\n')
        f.write('};\n')
    else:
        raise ValueError('shape is unvalid')
    f.close()

if __name__ == "__main__":
    text = 'hybrid'
    weight(f'../HLS/{text}/u.h', 'int', 'U', [len(np.load('../HLS/proposed_16bit/U.npy'))], '../HLS/proposed_16bit/U.npy')
    weight(f'../HLS/{text}/win.h', 'AP_RES', 'Win', [len(np.load('../HLS/proposed_16bit/Win.npy'))])
    weight(f'../HLS/{text}/x.h', 'AP_INT', 'X', [len(np.load('../HLS/proposed_16bit/X.npy'))])
    weight(f'../HLS/{text}/wout.h', 'AP_OUT', 'Wout', [len(np.load('../HLS/proposed_16bit/Wout.npy'))])
    weight(f'../HLS/{text}/d.h', 'int', 'D', [len(np.load('../HLS/proposed_16bit/D.npy'))], '../HLS/proposed_16bit/D.npy')