def mac_matmul(n, p, m):
    return n * p * m

def linear_layer_mac(seq_len, in_dim, out_dim):
    return seq_len * in_dim * out_dim


def conv_layer_mac(in_h, in_w, in_dim, out_dim, kernel_size, stride, padding):
    conv_ops = (in_h+padding*2) * (in_w+padding*2) // stride
    filter_mac = kernel_size**2 * in_dim * out_dim
    return 2 * conv_ops * filter_mac


def mac_for_lin_bax(seq_len, in_dim, out_dim, rank):
    return (
        linear_layer_mac(seq_len, in_dim, rank)
        + linear_layer_mac(seq_len, rank, out_dim)
    )


def mac_for_conv_bax(in_h, in_w, in_dim, out_dim, rank):
    return (
        conv_layer_mac(in_h, in_w, in_dim, rank, 3, 1, 1) 
        + conv_layer_mac(in_h, in_w, rank, out_dim, 1, 1, 0)
    )


def mac_for_lin_wba(in_dim, out_dim, rank):
    return mac_matmul(in_dim, rank, out_dim)


def mac_for_conv_wba(in_dim, out_dim, rank):
    return mac_matmul(in_dim, rank, out_dim * 3**2)


# shape list for 64x64 input of SD Unet
shape_list = [
    (320, 96, 96),
    (640, 48, 48),
    (1280, 24, 24),
    (1280, 12, 12)
]

mac_lin_bax = 0
mac_conv_bax = 0
mac_lin_wba = 0
mac_conv_wba = 0
rank = 128


for shape in shape_list:
    print(shape)
    original_lin = linear_layer_mac(shape[1]*shape[2], shape[0], shape[0])
    original_conv = conv_layer_mac(shape[1], shape[2], shape[0], shape[0], 3, 1, 1)
    
    mac_lin_bax = mac_for_lin_bax(shape[1]*shape[2], shape[0], shape[0], rank)
    mac_lin_backward_bax = (
        mac_matmul(rank, shape[1]*shape[2], shape[0])   # dB = dY*(AX)
        + mac_matmul(shape[1]*shape[2], shape[0], rank) # d(AX) = B*dY
        + mac_matmul(rank, shape[1]*shape[2], shape[0]) # dA = d(AX)*X
        + mac_matmul(shape[1]*shape[2], rank, shape[0]) # dX = A*d(AX)
    )
    mac_lin_wba = mac_for_lin_wba(shape[0], shape[0], rank)
    mac_lin_backward_wba = mac_matmul(rank, shape[0], shape[0])*2
    
    mac_conv_bax = mac_for_conv_bax(shape[1], shape[2], shape[0], shape[0], rank)
    mac_conv_backward_bax = (
        
    )# 2 times of forward I guess
    mac_conv_wba = mac_for_conv_wba(shape[0], shape[0], rank)
    mac_conv_backward_wba = mac_matmul(rank, shape[0]*3**2, shape[0]) + mac_matmul(rank, shape[0], shape[0]*3**2)

    print(f"lin")
    print(f"original_lin  : {original_lin}")
    print(f"mac_lin_bax   : {mac_lin_bax}")
    print(f"mac_lin_bwax  : {mac_lin_backward_bax}")
    print(f"mac_lin_wba   : {mac_lin_wba}")
    print(f"mac_lin_bwba  : {mac_lin_backward_wba}")
    print(f"conv")
    print(f"original_conv : {original_conv}")
    print(f"mac_conv_bax  : {mac_conv_bax}")
    print(f"mac_conv_wba  : {mac_conv_wba}")
    print(f"mac_conv_bwba : {mac_conv_backward_wba}")
    print()