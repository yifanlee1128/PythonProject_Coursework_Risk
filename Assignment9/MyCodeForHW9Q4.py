import numpy as np
import matplotlib.pyplot as plt

result=[ np.max(np.random.standard_t(5,10)) for i in range(1000)]
plt.hist(result,bins=40)
plt.title("Histogram of "+str(10)+"-Student t sample maxima, "+str(1000)+" trials")
plt.xlabel("Sample maximum")
plt.ylabel("Frequency")
plt.savefig("Histogram for HW9Q4.pdf")
plt.show()