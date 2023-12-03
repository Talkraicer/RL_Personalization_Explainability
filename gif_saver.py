from matplotlib import animation
import matplotlib.pyplot as plt
import os

"""
Ensure you have imagemagick installed with 
sudo apt-get install imagemagick
Open file in CLI with:
xgd-open <filelname>
"""
rep_index = 0

def action_to_string(action):
    if action == 0:
        return "LEFT"
    elif action == 1:
        return "IDLE"
    elif action == 2:
        return "RIGHT"
    elif action == 3:
        return "FASTER"
    elif action == 4:
        return "SLOWER"
    else:
        return "V"

def save_frames_as_gif(frames,speeds, actions, user = None , path='./', filename=''):

    actions = [action_to_string(action) for action in actions]

    #Mess with this to change frame size
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis('off')
    text = plt.text(0,0, f"", fontsize=10, color='red')
    # another text on top right corner
    text2 = plt.text(60,0, f"", fontsize=10, color='blue')

    # text of action
    text3 = plt.text(300, 0, f"", fontsize=10, color='green')

    if user:
        text4 = plt.text(400, 0, f"", fontsize=10, color='purple')
        user = [action_to_string(action) for action in user]


    def animate(i):
        global rep_index
        patch.set_data(frames[i])
        if i == 0:
            rep_index += 1
        text.set_text(f"Frame: {i}")
        text2.set_text(f"Speed: {speeds[i]}")
        text3.set_text(f"Action: {actions[i]}")
        if user:
            text4.set_text(f"User: {user[i]}")

    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames))

    # create folder if it doesn't exist
    if not os.path.exists(path):
        os.makedirs(path)

    anim.save(path + filename, writer='imagemagick', fps=5)

    plt.close()