from matplotlib import pyplot as plt

def ShowCount(weights, loss_sum_count):
    plt.subplot(2, 1, 1)
    plt.plot(range(0, 2000), weights, color='red')
    plt.xlabel('TimeSteps')
    plt.ylabel('Probability')

    plt.subplot(2, 1, 2)
    plt.plot(range(0, 2000), loss_sum_count, color='red')
    plt.xlabel('TimeSteps')
    plt.ylabel('Counts')
    plt.tight_layout()
    plt.savefig('Count_and_Probability.png')
    plt.close()
