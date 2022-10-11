from math import *
from sympy import *
import sympy
import numpy as np

#在我们调用Newton类的构造器创建对象时，首先会在内存中获得保存Newton对象所需的内存空间，
# 然后通过自动执行__init__方法，完成对内存的初始化操作，也就是把数据放到内存空间中。
# 所以我们可以通过给Newton类添加__init__方法的方式为Newton对象指定属性，
# 同时完成对属性赋初始值的操作，正因如此，__init__方法通常也被称为初始化方法。
#给Newton对象添加equations（方程）属性
class Newton:
    def __init__(self, equations):
        #初始化
        self.equations = equations
        self.n = len(equations)
        #.format()函数：替换前面花括号的内容
        #若n = 3则self.x为(x0, x1, x2, x3....)
        self.x = sympy.symbols(" ".join("x{}".format(i) for i in range(self.n)) + " x{}".format(self.n), real=True)
        #得到方程组的每个方程
        self.equationsSymbol = [equations[i](self.x) for i in range(self.n)]
        # 初始化 Jacobian 矩阵
        #dtype为数据类型，sympy.core.add.Add类型的对象，是一个表达式。定义了Jacobian矩阵的每个数据类型为表达式类型
        #reshape为形状参数是必须的
        self.J = np.zeros(self.n * self.n, dtype=sympy.core.add.Add).reshape(self.n, self.n)
        for i in range(self.n):
            for j in range(self.n):
                #可以通过使用sympy.diff()方法，得到带有变量的数学表达式的微分(求导)
                self.J[i][j] = sympy.diff(self.equationsSymbol[i], self.x[j])
    #计算Jacobian 矩阵
    def cal_J(self, x):
        dict = {self.x[i]: x[i] for i in range(self.n)}
        #生成一个n行n列的0矩阵
        J = np.zeros(self.n * self.n).reshape(self.n, self.n)
        for i in range(self.n):
            for j in range(self.n):
                #用x[i]替换self.x[i]
                J[i][j] = self.J[i][j].subs(dict)
        return J
    #构建方程表达式
    def cal_f(self, x):
        f = np.zeros(self.n)
        for i in range(self.n):
            f[i] = self.equations[i](x)
        f.reshape(self.n, 1)
        return f
    #定义牛顿法的初始值和迭代次数
    def interationsByStep(self, x0, step):
        x0 = np.array(x0)
        for i in range(step):
            #注意：这里的@不是修饰器，而是矩阵相乘的符号
            x0 = x0 - np.linalg.pinv(self.cal_J(x0)) @ self.cal_f(x0)
            print("Step {}:".format(i + 1), ", ".join(["x{} = {}".format(j + 1, x0[j]) for j in range(self.n)]))
        return x0

    # 定义牛顿法的初始值和计算精度
    def interationsByEpsilon(self, x0, epsilon):
        error = float("inf")
        while error >= epsilon:
            cal = np.linalg.pinv(self.cal_J(x0)) @ self.cal_f(x0)
            error = max(abs(cal))
            x0 = x0 - cal
        print(x0)
        return x0

#使用实例
if __name__ == "__main__":
    # equations 为方程组
    # x0 为初始值 step 为迭代次数 epsilon 为精度
    # 多元非线性使用方法
    equations = [lambda x: cos(0.4 * x[1] + x[0] ** 2) + x[0] ** 2 + x[1] ** 2 - 1.6,
                 lambda x: 1.5 * x[0] ** 2 - x[1] ** 2 / 0.36 - 1,
                 lambda x: 3 * x[0] + 4 * x[1] + 5 * x[2]]

    newton = Newton(equations)
    newton.interationsByStep([1, 1, 1], 5)
    newton.interationsByEpsilon([1, 1, 1], 0.001)

    # 一元非线性使用方法
    equations = [lambda x: cos(x[0]) + sin(x[0])]

    newton = Newton(equations)
    newton.interationsByStep([1], 5)
    newton.interationsByEpsilon([1], 0.001)