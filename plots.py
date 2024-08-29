import pandas as pd
import matplotlib.pyplot as plt

def plotContrast(left, right, leftName='', rightName='', title=''):
    # fp = hd.File('data.hd5', 'r')
    # walking = pd.DataFrame(fp['dataset/training/walking1'][:])
    walking = left
    walking = walking.dropna()
    walking.iloc[:, 0] -= walking.iloc[:, 0].min()
    # jumping = pd.DataFrame(fp['dataset/training/jumping1'][:])
    jumping = right
    jumping = jumping.dropna()
    jumping.iloc[:, 0] -= jumping.iloc[:, 0].min()

    # make the data the same length by trimming
    if(len(walking) > len(jumping)):
        walking = walking.iloc[:len(jumping), :]
    else:
        jumping = jumping.iloc[:len(walking), :]



    # acceleration plots of walking and jumping
    fig, ax = plt.subplots(ncols=2, nrows=3)
    fig.suptitle(title)
    ax.flatten()[0].plot(walking.iloc[:, 0], walking.iloc[:, 1], label='X')
    ax.flatten()[0].plot(walking.iloc[:, 0], walking.iloc[:, 2], label='Y')
    ax.flatten()[0].plot(walking.iloc[:, 0], walking.iloc[:, 3], label='Z')
    ax.flatten()[0].plot(walking.iloc[:, 0],  (walking.iloc[:, 1] +  walking.iloc[:, 2] + walking.iloc[:, 3])/3, label='Combined')
    ax.flatten()[0].set_title(leftName+' Acceleration')
    ax.flatten()[1].plot(jumping.iloc[:, 0], jumping.iloc[:, 1], label='X')
    ax.flatten()[1].plot(jumping.iloc[:, 0], jumping.iloc[:, 2], label='Y')
    ax.flatten()[1].plot(jumping.iloc[:, 0], jumping.iloc[:, 3], label='Z')
    ax.flatten()[1].plot(walking.iloc[:, 0], (jumping.iloc[:, 1] +  jumping.iloc[:, 2] + jumping.iloc[:, 3])/3, label='Combined')
    ax.flatten()[1].set_title(rightName+' Acceleration')

    # x, y, z velocity plots of walking and jumping
    xVel = walking.iloc[:, 1].cumsum()
    yVel = walking.iloc[:, 2].cumsum()
    zVel = walking.iloc[:, 3].cumsum()

    ax.flatten()[2].plot(walking.iloc[:, 0], xVel, label='X')
    ax.flatten()[2].plot(walking.iloc[:, 0], yVel, label='Y')
    ax.flatten()[2].plot(walking.iloc[:, 0], zVel, label='Z')
    ax.flatten()[2].plot(walking.iloc[:, 0], (xVel+yVel+zVel)/3, label='Combined')
    ax.flatten()[2].set_title(leftName+' Velocity')

    xVel = xVel.cumsum()
    yVel = yVel.cumsum()
    zVel = zVel.cumsum()

    ax.flatten()[4].plot(walking.iloc[:, 0], xVel, label='X')
    ax.flatten()[4].plot(walking.iloc[:, 0], yVel, label='Y')
    ax.flatten()[4].plot(walking.iloc[:, 0], zVel, label='Z')
    ax.flatten()[4].plot(walking.iloc[:, 0], (xVel+yVel+zVel)/3, label='Combined')
    ax.flatten()[4].set_title(leftName+' Displacement')


    xVel = jumping.iloc[:, 1].cumsum()
    yVel = jumping.iloc[:, 2].cumsum()
    zVel = jumping.iloc[:, 3].cumsum()

    ax.flatten()[3].plot(walking.iloc[:, 0], xVel, label='X')
    ax.flatten()[3].plot(walking.iloc[:, 0], yVel, label='Y')
    ax.flatten()[3].plot(walking.iloc[:, 0], zVel, label='Z')
    ax.flatten()[3].plot(walking.iloc[:, 0], (xVel+yVel+zVel)/3, label='Combined')
    ax.flatten()[3].set_title(rightName+' Velocity')

    xVel = xVel.cumsum()
    yVel = yVel.cumsum()
    zVel = zVel.cumsum()

    ax.flatten()[5].plot(walking.iloc[:, 0], xVel, label='X')
    ax.flatten()[5].plot(walking.iloc[:, 0], yVel, label='Y')
    ax.flatten()[5].plot(walking.iloc[:, 0], zVel, label='Z')
    ax.flatten()[5].plot(walking.iloc[:, 0], (xVel+yVel+zVel)/3, label='Combined')
    ax.flatten()[5].set_title(rightName+' Displacement')

    fig.tight_layout()
    plt.legend()
    plt.show()

# Plot the first 3 walking and jumping frames from the dataset
def plot3(df, title='Data'):
    walkingFrames = []
    jumpingFrames = []
    for frame in df:
        if(frame['label'].iloc[0] == 0 and len(walkingFrames) < 3):
            walkingFrames.append(frame)
        elif(frame['label'].iloc[0] == 1 and len(jumpingFrames) < 3):
            jumpingFrames.append(frame)

    for i in range(3):
        plotContrast(jumpingFrames[i], walkingFrames[i], title=f'{title} {i+1}', leftName='Jumping', rightName='Walking')




