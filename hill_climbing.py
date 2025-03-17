from mpi4py import MPI
import numpy as np
import time
import copy

def load_data(path = '/home/haopd/code/HPC/test/test10.txt'):
  with open(path, 'r') as f:
    inputData = f.readlines()
  N_K=inputData[0].strip().split(' ')
  N=int(N_K[0])
  K=int(N_K[1])
  dis_matrix = np.zeros([N+1,N+1])
  for i in range(0,N+1):
    tmp_data=inputData[i+1].strip()
    tmp_data=tmp_data.split(" ")
    for j in range(0,N+1):
      dis_matrix[i,j]=tmp_data[j]
  return N,K,dis_matrix

def init(K, N):
    # Khởi tạo ngẫu nhiên các tuyến (mỗi tuyến bắt đầu với depot 0)
    X = [[0] for _ in range(K)]
    for i in range(N):
        a = np.random.randint(0, K)
        X[a].append(i + 1)
    return X

def calculate_dis(X, dis_matrix, K):
    dis = np.zeros((K, 1))
    for i in range(K):
        len_Xi = len(X[i]) - 1
        for j in range(len_Xi):
            a = int(X[i][j])
            b = int(X[i][j + 1])
            dis[i] += dis_matrix[a][b]
        # Cộng khoảng cách từ điểm cuối về depot (0)
        c = int(X[i][len_Xi])
        dis[i] += dis_matrix[c][0]
    return dis

def getNeighbors(X, index_max, index_min):
    neighbors = []
    for i in range(len(X[index_max]) - 1):
        Y = copy.deepcopy(X)
        # Chọn vị trí ngẫu nhiên trong tuyến có chi phí thấp để chèn điểm
        a = np.random.randint(1, len(X[index_min]) + 1)
        tmp = X[index_max][i + 1]
        Y[index_max].remove(tmp)
        Y[index_min].insert(a, tmp)
        neighbors.append(Y)
    # print(neighbors)
    return neighbors

def hill_climbing_mpi(N, K, dis_matrix, max_iter=100):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    # print(size)

    # Tiến trình 0 khởi tạo giải pháp ban đầu

    best_X = init(K, N)
    best_dis = calculate_dis(best_X, dis_matrix, K)
    


        # print(best_dis)
        # best_dis = comm.recv(source=0)

    # Phát broadcast giải pháp ban đầu cho tất cả tiến trình
    # best_X = comm.bcast(best_X, root=0)
    # best_dis = comm.bcast(best_dis, root=0)

    for _ in range(max_iter):
        current_dis = calculate_dis(best_X, dis_matrix, K)
        index_max = int(np.argmax(current_dis))
        index_min = int(np.argmin(current_dis))

        # Tiến trình 0 tạo các giải pháp hàng xóm
        NeighborX = getNeighbors(best_X, index_max, index_min)     
        for neighbor in NeighborX:
            neighbor_dis = calculate_dis(neighbor, dis_matrix, K)
            max_neighbor_dis = max(neighbor_dis)
            if max_neighbor_dis < max(best_dis):
                best_X = neighbor
                best_dis = neighbor_dis
            
    best_Xs = comm.allgather(best_X)
    best_dises = comm.allgather(best_dis)
    if rank == 0:
        max_value_dises = np.array([max(dis) for dis in best_dises]) # lấy đường đi dài nhất của bưu tá tất cả tiến trình
        best_index = np.argmin(max_value_dises)
        print(len(best_Xs))
        print(best_dises)
        return best_Xs[best_index], max_value_dises[best_index]
    else:
        return None, None

if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    # Ví dụ: sử dụng N = 10, K = 3, và ma trận khoảng cách ngẫu nhiên
    N,K,dis_matrix=load_data()
    ti = time.time()
    best_X, best_cost = hill_climbing_mpi(N, K, dis_matrix, max_iter=100)
    
    if rank == 0:
        print("Best solution:", best_X)
        print("Best cost:", best_cost)
        print(time.time()-ti)
