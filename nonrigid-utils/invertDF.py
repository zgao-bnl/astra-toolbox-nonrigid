import cupy as cp
def invertDF(dx,dy,dz,it = 10):
    source=r'''
    extern "C"{
    __global__ void invertDF(float* drX, float* drY, float* drZ,
                               cudaTextureObject_t texDx,
                               cudaTextureObject_t texDy,
                               cudaTextureObject_t texDz,
                               int xdim, int ydim, int zdim, int iter)
    {
        unsigned int xi = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned int yi = blockIdx.y * blockDim.y + threadIdx.y;
        unsigned int zi = blockIdx.z * blockDim.z + threadIdx.z;
        
        unsigned int coord = (zi*ydim+yi)*xdim+xi;


        // Read from texture and write to global memory
        float x = xi+0.5f;
        float y = yi+0.5f;
        float z = zi+0.5f;
        
        float xd = 0.0f;
        float yd = 0.0f;
        float zd = 0.0f;
        
        int i = 0;

        if (x < xdim && y < ydim && z < zdim) {
            float vx = - tex3D<float>(texDx,x,y,z);
            float vy = - tex3D<float>(texDy,x,y,z);
            float vz = - tex3D<float>(texDz,x,y,z);

            float vx1 = 0.0f;
            float vy1 = 0.0f;
            float vz1 = 0.0f;
            for (int i=0; i<iter; ++i) {
                xd = x - vx;
                yd = y - vy;
                zd = z - vz;

                vx1 = - tex3D<float>(texDx,xd,yd,zd);
                vy1 = - tex3D<float>(texDy,xd,yd,zd);
                vz1 = - tex3D<float>(texDz,xd,yd,zd);

                vx = vx*0.35+vx1*0.65;
                vy = vy*0.35+vy1*0.65;
                vz = vz*0.35+vz1*0.65;
            }

            drX[coord] = vx;
            drY[coord] = vy;
            drZ[coord] = vz;
        }
    }
    }
    '''
    dims = dx.shape

    ch = cp.cuda.texture.ChannelFormatDescriptor(32, 0, 0, 0, cp.cuda.runtime.cudaChannelFormatKindFloat)
    arrX = cp.cuda.texture.CUDAarray(ch, dims[0], dims[1], dims[2])
    arrY = cp.cuda.texture.CUDAarray(ch, dims[0], dims[1], dims[2])
    arrZ = cp.cuda.texture.CUDAarray(ch, dims[0], dims[1], dims[2])

    resX = cp.cuda.texture.ResourceDescriptor(cp.cuda.runtime.cudaResourceTypeArray, cuArr=arrX)
    resY = cp.cuda.texture.ResourceDescriptor(cp.cuda.runtime.cudaResourceTypeArray, cuArr=arrY)
    resZ = cp.cuda.texture.ResourceDescriptor(cp.cuda.runtime.cudaResourceTypeArray, cuArr=arrZ)

    tex_des = cp.cuda.texture.TextureDescriptor((cp.cuda.runtime.cudaAddressModeClamp, cp.cuda.runtime.cudaAddressModeClamp, cp.cuda.runtime.cudaAddressModeClamp),
                                            cp.cuda.runtime.cudaFilterModeLinear,
                                            cp.cuda.runtime.cudaReadModeElementType)

    texDx = cp.cuda.texture.TextureObject(resX, tex_des)
    texDy = cp.cuda.texture.TextureObject(resY, tex_des)
    texDz = cp.cuda.texture.TextureObject(resZ, tex_des)

    # allocate input/output arrays
    tex_dataX = cp.array(dx,dtype=cp.float32)
    tex_dataY = cp.array(dy,dtype=cp.float32)
    tex_dataZ = cp.array(dz,dtype=cp.float32)
    
    drX = cp.zeros_like(tex_dataX)
    drY = cp.zeros_like(tex_dataY)
    drZ = cp.zeros_like(tex_dataZ)

    arrX.copy_from(tex_dataX)
    arrY.copy_from(tex_dataY)
    arrZ.copy_from(tex_dataZ)

    # get the kernel, which copies from texture memory
    ker = cp.RawKernel(source, 'invertDF')

    # launch it
    block_x = 4
    block_y = 4
    block_z = 4
    grid_x = (dims[0] + block_x - 1)//block_x
    grid_y = (dims[1] + block_y - 1)//block_y
    grid_z = (dims[2] + block_z - 1)//block_z

    ker((grid_x, grid_y, grid_z), (block_x, block_y, block_z), (drX,drY,drZ,texDx,texDy,texDz,dims[0],dims[1],dims[2],it))
    
    return drX.get(),drY.get(),drZ.get()

