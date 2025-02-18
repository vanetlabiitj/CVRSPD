import numpy as np
#import matplotlib.pyplot as plt
from scipy.spatial import distance
import random
import pulp
import gurobipy as gp
from gurobipy import GRB
from scipy.spatial import distance

class VRPSDP_GUROBI:
	def __init__(self, costMatrix, demand, pickup, numberOfVehicles, capacityOfVehicle):
		self.costMatrix = costMatrix
		self.n = len(costMatrix)
		self.demand = demand
		self.pickup = pickup
		self.numberOfVehicles = numberOfVehicles
		self.capacityOfVehicle = capacityOfVehicle
		self.initialzeLP()

	def initialzeLP(self):
		self.cvrpLP = gp.Model('VRP-SDP')
		x, R, P = [], [], []

		# Create decision variables
		for i in range(self.n):
			xRow, RRow, PRow = [], [], []
			for j in range(self.n):
				xRow.append(self.cvrpLP.addVar(name='x('+str(i)+","+str(j)+")", vtype=GRB.BINARY, lb=0, ub=1))
				RRow.append(self.cvrpLP.addVar(name='R('+str(i)+","+str(j)+")", vtype=GRB.INTEGER, lb=0))
				PRow.append(self.cvrpLP.addVar(name='P('+str(i)+","+str(j)+")", vtype=GRB.INTEGER, lb=0))
			x.append(xRow)
			R.append(RRow)
			P.append(PRow)

		# Create objective
		objective = None
		for i in range(self.n):
			for j in range(self.n):
				objective += self.costMatrix[i][j] * x[i][j]
		self.cvrpLP.setObjective(objective,GRB.MINIMIZE)

		# constraint 1
		for j in range(1, self.n):
			const1 = None
			for i in range(self.n):
				if(const1 == None):
					const1 = x[i][j]
				else:
					const1 = const1 + x[i][j]
			self.cvrpLP.addConstr(const1 == 1)
		
		# constraint 2
		for j in range(1, self.n):
			const2 = None
			for i in range(self.n):
				if(const2 == None):
					const2 = x[j][i]
				else:
					const2 = const2 + x[j][i]
			self.cvrpLP.addConstr(const2 == 1)

		# constraint 3
		for j in range(1, self.n):
			const3a, const3b = None, None
			for i in range(self.n):
				if(const3a == None):
					const3a = R[i][j]
				else:
					const3a = const3a + R[i][j]
				if(const3b == None):
					const3b = R[j][i]
				else:
					const3b = const3b + R[j][i]
			self.cvrpLP.addConstr(const3a - self.demand[j] == const3b)

		# constraint 4
		for j in range(1, self.n):
			const4a, const4b = None, None
			for i in range(self.n):
				if(const4a == None):
					const4a = P[i][j]
				else:
					const4a = const4a + P[i][j]
				if(const4b == None):
					const4b = P[j][i]
				else:
					const4b = const4b + P[j][i]
			self.cvrpLP.addConstr(const4a + self.pickup[j] == const4b)

		# constraint 5
		const5 = None
		for i in range(1, self.n):
			if(const5 == None):
				const5 = P[0][i]
			else:
				const5 = const5 + P[0][i]
		self.cvrpLP.addConstr(const5 == 0)

		# constraint 6
		const6 = None
		for i in range(1, self.n):
			if(const6 == None):
				const6 = R[i][0]
			else:
				const6 = const6 + R[i][0]
		self.cvrpLP.addConstr(const6 == 0)

		# constraint 7
		for i in range(self.n):
			for j in range(self.n):
				self.cvrpLP.addConstr(R[i][j] + P[i][j] <= self.capacityOfVehicle * x[i][j])

		# constraint 8
		for i in range(1, self.n):
			self.cvrpLP.addConstr(x[0][i] <= self.numberOfVehicles)

	def solve(self):
		status = self.cvrpLP.optimize()
		print(status)
	
	def getResult(self):
		print("Objective value: ", self.cvrpLP.ObjVal)
		for v in self.cvrpLP.getVars():
			print(v.varName, " = ", v.x)
		return self.cvrpLP

#adding AD and RD

def VRPsolutiontoList(sol):
    firsts = np.where(sol[0]==1)[0]
    solution = []
    
    for f in range(len(firsts)):
        route = [0]
        # print("Vehicle {}".format(f+1))
        
        dest = 100
        source = firsts[f]
        # print("Vehicle {} goes from Depot to Stop Index {}".format(f+1,source))
        
        while dest !=0:
            route.append(source)
            dest = np.argmax(sol[source])


            # if source==0:
            #     print("Vehicle {} goes from Stop index {} to Stop Index {}".format(f+1,source,dest))
            # elif dest==0:
            #     print("And finally, Vehicle {} goes from Stop index {} to Depot".format(f+1,source))
            # else:
            #     print("Then, Vehicle {} goes  from Stop index {} to Stop Index {} ".format(f+1, source,dest))
            source = dest
        route.append(0)
        solution.append(route)
        # print("______________________")
    return solution

# arc difference
def eval_ad(P, A):
    P_set = set()
    for sublist in P:
        for i in range(len(sublist)-1):
            P_set.add( (sublist[i],sublist[i+1]) )
        
    A_set = set()
    for sublist in A:
        for i in range(len(sublist)-1):
            A_set.add( (sublist[i],sublist[i+1]) )
    
    result = set(A_set).difference((set(P_set)))
    
    # assert(len(A_set) == len(P_set))
    # diffset, diffcount, diffrelative
    return  len(result), len(result)/len(A_set)


# locate minimum
# save location in a list (idx_list)
# before next iteration, increase all values in row and column where minimum is located by a large number
def get_best_route_mapping(P, A):
    # create initial matrix of symmetric differences
    np_matrix = np.zeros((len(P),len(A)))
    for x in range(len(P)):
        for y in range(len(A)):
            np_matrix[x][y] = len(set(P[x]).symmetric_difference(set(A[y])))

    idx_list = []
    while len(idx_list) < len(A):
        # find smallest (x,y) in matrix
        (idx_r, idx_c) = np.where(np_matrix == np.nanmin(np_matrix))
        (r, c) = (idx_r[0], idx_c[0]) # avoid duplicates
        idx_list.append( (r, c) )

        # blank out row/column selected
        np_matrix[r,:] = np.NaN
        np_matrix[:,c] = np.NaN
        #print(np_matrix)
        #print(len(idx_list), len(Act))
    
    return idx_list

# get all unique stops
def allstops(R):
    result = set()
    for route in R:
        result.update(route)
    return result

# stop difference
def eval_sd(P, A):
    # get paired mapping
    idx_list = get_best_route_mapping(P, A)
    
    diff = set()
    for (idx_P, idx_A) in idx_list:
        diff.update(set(P[idx_P]).symmetric_difference(set(A[idx_A])))
    
    nr_stops = len(allstops(A))
    # diffset, diffcount, diffrelative
    return  len(diff), len(diff)/nr_stops
'''
costMatrix = [[0,9,14,23,32,50,21,49,30,27,35,28,18],   
[9,0,21,22,36,52,24,51,36,37,41,30,20],    
[14,21,0,25,38,5,31,7,36,43,29,7,6],    
[23,22,25,0,42,12,35,17,44,31,31,11,6],
[32,36,38,42,0,22,37,16,46,37,29,13,14],   
[50,52,5,12,22,0,41,23,10,39,9,17,16],   
[21,24,31,35,37,41,0,26,21,19,10,25,12],  
[49,51,7,17,16,23,26,0,30,28,16,27,12],   
[30,36,36,44,46,10,21,30,0,25,22,10,20],    
[27,37,43,31,37,39,19,28,25,0,20,16,8],   
[35,41,29,31,29,9,10,16,22,20,0,10,10],   
[28,30,7,11,13,17,25,27,10,16,10,0,10],
[18,20, 6, 6,14,16,12,12,20,8, 10,10,0]]'''

'''
xCoordinates = [145, 151, 159, 130, 128, 163, 146, 161, 142, 163, 148, 128, 156, 129, 146, 164, 141, 147, 164, 129, 155, 139]
yCoordinates = [215, 164, 261, 254, 252, 247, 246, 242, 239, 236, 232, 231, 217, 214, 208, 208, 206, 193, 193, 189, 185, 182]
costMatrix = np.ndarray(shape=(len(xCoordinates), len(yCoordinates)))
'''
