import cv2
import numpy as np

# threshold : 합성 기준선(-1~1)
# offset : 합성 기준선의 너비(0 이상 알잘딱 값)
# direction : 합성 방향(0: 왼쪽, 1: 오른쪽)
def tex_interp( uvtex, threshold=0, offset=50, direction=1 ): 
    origin_tex = uvtex.copy()
    flip_tex = cv2.flip(origin_tex, 1).copy()
    
    border = int((uvtex.shape[1]/2)*(threshold+1))
    
    mask = np.zeros([uvtex.shape[0], uvtex.shape[1], 3], dtype=np.float32)
    max_val = 1
    mask[:, :(border), :] = max_val
    for x in range(border-offset, border):
        mask[:, x, :] = max_val*(abs((x-border)/offset))
    mask_flip = max_val-mask
    
    result = np.zeros([uvtex.shape[0], uvtex.shape[1], 3], dtype=np.float32)
    if(direction):
        result = (origin_tex*mask_flip + flip_tex*mask) / max_val
    else:
        result = (origin_tex*mask + flip_tex*mask_flip) / max_val

    
    return result.astype(np.uint8)
    
 
 
#test
if __name__ == '__main__':
    img = cv2.imread('../data/uvtex1.png', cv2.IMREAD_COLOR)
    mask = tex_interp( img, 0, 30, 0)
    cv2.imshow('img', img)
    cv2.imshow('mask', mask)
    cv2.waitKey(0)