def signed_binary_to_decimal(val:str):
    return - (int(val,2) & int('1'+'0'*(len(val)-1),2)) | \
        (int(val,2) & int('0'+'1'*(len(val)-1),2))

def decimal_to_signed_binary(val:int, n:int):
    if(val>=2**(n-1)):
        return '0'+'1'*(n-1)
    elif(val<-2**(n-1)):
        return '1'+'0'*(n-1)
    else:
        if(val>=0):
            return str(format(val,'0'+str(n)+'b'))
        else:
            return str(format(val&int('1'*n,2),'0'+str(n)+'b'))

def exor(val1:str, val2:str):
    if(len(val1)!=len(val2)):
        raise ValueError('bit width must be the same')
    else:
        return str(format(int(val1,2)^int(val2,2), '0'+str(len(val1))+'b'))

def approx_in_signed_binary(val:float, n:int):
    if(val>=2**(n-1)):
        return 2**(n-1)-1
    elif(val<-2**(n-1)):
        return -2**(n-1)
    else:
        return int(val)
    
def right_shift(val:int, n:int, bit:int):
    val_bin = decimal_to_signed_binary(val,bit)
    val_bin = val_bin[0]*n + val_bin[0:bit-n]
    return signed_binary_to_decimal(val_bin)