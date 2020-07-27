
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats

x = np.linspace(0,10,100)
sigma_x = np.std(x)
y = x
sigma_y = np.std(y)
cov = np.cov(x,y,bias=True)[0][1]
r = cov/(sigma_x*sigma_y)
# r = scipy.stats.pearsonr(x,y)[0]
r_2 = r**2
print(r_2)

plt.scatter(x,y,label='$r^2$='+str(round(r_2,4)))

# add gaussian noise
noise1 = np.random.normal(0,1,100)
y1 = y + noise1
sigma_y1 = np.std(y1)
cov1 = np.cov(x,y1,bias=True)[0][1]
r1 = cov1/(sigma_x*sigma_y1)
# r1 = scipy.stats.pearsonr(x,y1)[0]
r1_2 = r1**2
print(r1_2)
plt.scatter(x,y1,label='$r^2$='+str(round(r1_2,4)))

# add gaussian noise with more variance
noise2 = np.random.normal(0,2,100)
y2 = y + noise2
sigma_y2 = np.std(y2)
cov2 = np.cov(x,y1,bias=True)[0][1]
r2 = cov2/(sigma_x*sigma_y2)
# r2 = scipy.stats.pearsonr(x,y2)[0]
r2_2 = r2**2
print(r2_2)
plt.scatter(x,y2,label='$r^2$='+str(round(r2_2,4)))

plt.title('Linear Data')
plt.legend()
plt.grid()

plt.ylim([-5,15])

plt.show()

