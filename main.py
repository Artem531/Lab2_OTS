import numpy as np
import scipy.special
from scipy.stats import rayleigh
import matplotlib.pyplot as plt

f0 = 40e6
T = 1e-6
delta = 1 / T
q = 16

fi_list = []
for i in range(q):
    fi_list.append(f0 + delta * i)

sec_num = 10000
Ts = 8 * T
E = 2
time = np.linspace(-Ts, Ts, sec_num)
signals_list = []
for f in fi_list:
    signals_list.append(np.sqrt(2 * E / T) * np.cos(2*np.pi * f * time))
signals_list = np.array(signals_list)

Maxdb = 30
SNRdb = list(range(0, Maxdb, 1))
num_thetas = 100
thetas = np.linspace(0, 2 * np.pi, num_thetas)

def get_ci(r, T, f, time):
    return np.sum(r * np.sqrt(2/T) * np.cos(2 * np.pi * f * time) * 1/sec_num)

def get_si(r, T, f, time):
    return np.sum(r * np.sqrt(2/T) * np.sin(2 * np.pi * f * time) * 1/sec_num)

Pe = []

e = 0
u_sigma = np.sqrt((1-e)/2)
u_mean = np.sqrt(e/2)

for db in SNRdb:
    SNR = 10 ** (db / 10)
    Nerr_max = 100
    Nerr = 0
    Ntest = 0
    sigma = np.sqrt(1/(2*SNR*q) * (np.sum( signals_list * signals_list )))
    #print(sigma)
    while(Nerr < Nerr_max):
        signal_idx = np.random.choice(q, 1)[0]
        f = fi_list[signal_idx]
        theta_idx = np.random.choice(num_thetas, 1)[0]
        theta = thetas[theta_idx]
        signal = np.sqrt(2 * E / T) * np.cos(2*np.pi * f * time - theta)

        x = np.random.normal(u_mean,u_sigma, 1)
        y = np.random.normal(u_mean,u_sigma, 1)
        u = np.sqrt(x**2 + y**2 )
        r = signal * u + np.random.normal(0, sigma, sec_num)

        ci_list = np.array([get_ci(r, T, fi, time) for fi in fi_list])
        si_list = np.array([get_si(r, T, fi, time) for fi in fi_list])

        res = np.argmax(ci_list**2 + si_list**2)
        #print(res == signal_idx)
        print(Nerr, db)
        if res != signal_idx:
            Nerr += 1
        Ntest += 1
    Pe.append(Nerr/Ntest)

print(Pe)

SNRdbtheory = np.linspace(0, Maxdb + 1, 100)
SNR = 10**(SNRdbtheory / 10)
UpperPe = (q - 1) / 2 * np.exp(-SNR/2)

accPe = [np.sum([ scipy.special.binom(q-1, l)*(-1)**(l+1) * (1+l+l*(1-e)*snr)**(-1) * np.exp(-( (l*e*snr)/(1+l+l*(1-e)*snr) )) for l in range(1, q)]) for snr in SNR]
print(accPe)
fig = plt.figure()
ax1= fig.add_subplot(1,1,1)
ax1.set_yscale("log")
ax1.set_ylabel('Pe')
ax1.set_xlabel('SNRdb')

line1, = plt.plot(SNRdb, Pe, label="My Pe", linestyle='--')
#line2, = plt.plot(SNRdbtheory, UpperPe, label="Upper", linestyle=':')
line3, = plt.plot(SNRdbtheory, accPe, label="Acc Pe", linestyle=':')

plt.legend(loc='upper right', borderaxespad=0.)
plt.show()

# fig, _ = plt.subplots()
# plt.plot(time[:100], signals_list[0][:100])
# plt.show()
#print(signals_list)




