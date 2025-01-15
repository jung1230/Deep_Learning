import matplotlib.pyplot as plt

print("----------------------- Task 2.1 -----------------------")
#  Task 2.1 
class BioModel(object):
    def __init__(self, sequence):
        self.sequence = sequence


    # Task 2.4
    def __iter__(self):
        # Initialize the index
        self.index =-1 

        return self
    def __next__(self):
        self.index += 1
        if self.index < len(self.sequence):
            return self.sequence[self.index]
        else:
            raise StopIteration


    # Task 2.6
    def __eq__(self, other):
        if len(self.sequence) == len(other.sequence):
            return sum(1 for a, b in zip(self.sequence, other.sequence) if a == b)
        else:
            raise ValueError("Two arrays are not equal in length!")
                
print("----------------------- Task 2.2 -----------------------")
# Task 2.2
class ExponentialGrowthModel(BioModel):
    def __init__(self, start, rate):
        # remember to init superclass as well!!! super(SubclassName, self).__init__() is for python 2, in 3 simply use super().__init__()
        super().__init__([]) #initialize BioModel with empty sequence 

        self.start = start
        self.rate = rate


    # Task 2.3
    def __call__(self, length):
        self.sequence = [self.start] 
        for i in range(length - 1):
            self.sequence.append(self.sequence[i] * (1+self.rate))
        
        print(self.sequence)
        return self.sequence
    
    def __len__(self):
        return(len(self.sequence))
    
GM = ExponentialGrowthModel(start=100, rate=0.1)
GM(length=5) # [100, 110, 121, 133.1, 146.41]
print(len(GM))

print("----------------------- Task 2.4 -----------------------")
# Task 2.4
print([n for n in GM]) 

print("----------------------- Task 2.5 -----------------------")
# Task 2.5
class ExponentialDecayModel(BioModel):
    def __init__(self, start, rate):
        # init superclass
        super().__init__([])

        self.start = start
        self.rate = rate

    def __call__(self, length):
        self.sequence = [self.start] 

        for i in range(length - 1):
            self.sequence.append(self.sequence[i] * (1-self.rate))
        
        print(self.sequence)
        return self.sequence
    
    def __len__(self):
        return(len(self.sequence))
    
DM = ExponentialDecayModel(start=100, rate=0.2)
DM(length=5) # [100, 80, 64, 51.2, 40.96]
print(len(DM)) # 5
print([n for n in DM]) # [100, 80, 64, 51.2, 40.96]

print("----------------------- Task 2.6 -----------------------")
# Task 2.6
GM = ExponentialGrowthModel(start=100, rate=0.1)
GM(length=5) # [100, 110, 121, 133.1, 146.41]

GM2 = ExponentialGrowthModel(start=100, rate=0.2)
GM2(length=5) # [100, 120, 144, 172.8, 207.36]

print(GM == GM2) # 1


GM3 = ExponentialGrowthModel(start=100, rate=0.2)
GM3(length=3) # [100, 120, 144]

# print(GM == GM3) # will raise an error

print("----------------------- Task 2.7 -----------------------")
# Task 2.7
GM = ExponentialGrowthModel(start=200, rate=0.9)
GM(length=5) 
GM2 = ExponentialDecayModel(start=400, rate=0.1)
GM2(length=5) 
print(len(GM)) # 5
print([n for n in GM]) 
print(len(GM2)) # 5
print([n for n in GM2]) 
print(GM == GM2) # 0
print(GM == GM) # 5

GM3 = ExponentialDecayModel(start=100, rate=0.2)
GM3(length=3)
# print(GM == GM3) # will raise an error

print("----------------------- Task 3.1 -----------------------")
# Task 3.1
class CombinedBioModel(BioModel):
    def __init__(self, growth_start, growth_rate, decay_start, decay_rate):
        super().__init__([])
        self.growth = ExponentialGrowthModel(growth_start, growth_rate)
        self.decay = ExponentialDecayModel(decay_start, decay_rate)
    
    def __call__(self, length):
        self.growth(length=length)
        self.decay(length=length)
        self.sequence = []
        for i in range(len(self.growth)):
            self.sequence.append(self.growth.sequence[i] * self.decay.sequence[i])
        print(self.sequence)
        return self.sequence
        
CBM = CombinedBioModel(growth_start=100, growth_rate=0.1, decay_start=1.0, decay_rate=0.05)
CBM(length=5) # [100.0, 104.50, 109.20, 114.12, 119.25]


print("----------------------- Task 3.2 -----------------------")
# Task 3.2
start1 = 10
start2 = 12
rate1 = 0.5
rate2 = 0.6
GM1 = ExponentialGrowthModel(start=start1, rate=rate1)
GM1(length=5) 
DM1 = ExponentialDecayModel(start=start1, rate=rate1)
DM1(length=5) 
CM1 = CombinedBioModel(growth_start=start1, growth_rate=rate1, decay_start=start1, decay_rate=rate1)
CM1(length=5) 
GM2 = ExponentialGrowthModel(start=start2, rate=rate2)
GM2(length=5) 
DM2 = ExponentialDecayModel(start=start2, rate=rate2)
DM2(length=5) 
CM2 = CombinedBioModel(growth_start=start2, growth_rate=rate2, decay_start=start2, decay_rate=rate2)
CM2(length=5) 

fig, axs = plt.subplots(2, 2, figsize=(10, 10))

axs[0, 0].plot(GM1.sequence, color="blue", label="start = 10, rate=0.5")
axs[0, 0].plot(DM1.sequence, color="red", label="start = 10, rate=0.5")
axs[0, 0].set_title("Blue Growth, Red Decay")
axs[0, 0].set_xlabel('Time')
axs[0, 0].set_ylabel('Value')
axs[0, 0].legend()
axs[0, 0].set_ylim(-10, 150)

axs[0, 1].plot(GM2.sequence, color="blue", label="start = 12, rate=0.6")
axs[0, 1].plot(DM2.sequence, color="red", label="start = 12, rate=0.6")
axs[0, 1].set_title("Blue Growth, Red Decay")
axs[0, 1].set_xlabel('Time')
axs[0, 1].set_ylabel('Value')
axs[0, 1].legend()
axs[0, 1].set_ylim(-10, 150)

axs[1, 0].plot(CM1.sequence, color="green", label="start = 10, rate=0.5")
axs[1, 0].set_title("Combined Sequence")
axs[1, 0].set_xlabel('Time')
axs[1, 0].set_ylabel('Value')
axs[1, 0].legend()
axs[1, 0].set_ylim(-10, 150)

axs[1, 1].plot(CM2.sequence, color="green", label="start = 12, rate=0.6")
axs[1, 1].set_title("Combined Sequence")
axs[1, 1].set_xlabel('Time')
axs[1, 1].set_ylabel('Value')
axs[1, 1].legend()
axs[1, 1].set_ylim(-10, 150)

plt.suptitle("Growth, Decay, and Combined for Different Rates")
plt.legend()


plt.show()
    