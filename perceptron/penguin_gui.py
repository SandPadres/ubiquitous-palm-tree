import tkinter as tk
import pandas as pd
from PIL import ImageTk, Image
import numpy as np

from file_path import PERCEPTRON_FILE_PATH
pengions = pd.read_csv(f"{PERCEPTRON_FILE_PATH}/penguins.csv") # read in penguins
pengions = pengions.drop(["rowid","island","sex","year"], axis=1).dropna() #select data columns, and drop NAs
pengions = pengions.loc[pengions["species"].isin(["Adelie","Gentoo"])] # Want 2 pengion species to binary classify
pengions = pengions.sample(frac=1) # Mix up the datapoints so not all Adelie, then all Gentoo
penguins_for_printing = pengions.copy() # Copy dataframe, as need a non-normalized version for displaying
pengions["bill_length_mm"] = (pengions["bill_length_mm"] - pengions["bill_length_mm"].mean() ) / pengions["bill_length_mm"].std() # standardize feature rows with mean 0 std 1 
pengions["bill_depth_mm"] = (pengions["bill_depth_mm"] - pengions["bill_depth_mm"].mean() ) / pengions["bill_depth_mm"].std()
pengions["flipper_length_mm"] = (pengions["flipper_length_mm"] - pengions["flipper_length_mm"].mean() ) / pengions["flipper_length_mm"].std()
pengions["body_mass_g"] = (pengions["body_mass_g"] - pengions["body_mass_g"].mean() ) / pengions["body_mass_g"].std()
labels = pengions["species"].to_numpy().reshape((1,len(pengions))) #y
labels = np.where(labels == "Adelie", -1, 1) # turn y values into 1, -1
data = pengions.drop(["species"], axis=1).to_numpy().T #x
data_display = penguins_for_printing.drop(["species"], axis=1).to_numpy().T

current_penguin_data = data[:,0]
current_penguin_data_display = data_display[:,0]
current_penguin_actual = labels[:, 0]
next_index = 1
# print(data[:,0])
# print(data[:,0][0])
# print(labels.shape)

# Load Classifer
penguin_th = np.load("penguin_th.npy")
penguin_th0 = np.load("penguin_th0.npy")
# penguin_th[0][0] = 0
# penguin_th[1][0] = 0
# penguin_th[2][0] = 0
# penguin_th[3][0] = 0
# penguin_th0[0][0] = 0
th_for_printing = np.array2string(penguin_th.ravel(), separator=',')
th0_for_printing = np.array2string(penguin_th0.ravel(), separator=',')

def hyperplane_side(x):
    return np.sign(penguin_th.T@x + penguin_th0)

def predict_penguin():
    global current_penguin_data
    global current_penguin_data_display
    global current_penguin_actual
    global next_index
    

    button_text = button.cget("text")
    if button_text == "Predict Penguin":
        
        # get prediction
        pred_y = hyperplane_side(current_penguin_data)
        # print((current_penguin_actual.item(), pred_y.item()))

        # Update images
        if current_penguin_actual.item() == -1:
            real_label.config(image=adelie_img)
            penguin_actual.config(text="Adelie")
        else:
            real_label.config(image=gentoo_img)
            penguin_actual.config(text="Gentoo")

        if pred_y.item() == -1:
            predicted_label.config(image=adelie_img)
            penguin_pred.config(text="Adelie")
        else:
            predicted_label.config(image=gentoo_img)
            penguin_pred.config(text="Gentoo")

        # Update Button text
        button.config(text="Next")


        if next_index < data.shape[1]:
            current_penguin_data = data[:,next_index]
            current_penguin_data_display = data_display[:,next_index]
            current_penguin_actual = labels[:, next_index]
            next_index = next_index + 1
        else:
            next_index = next_index + 1

    elif button_text == "Next":
        # Update Images
        real_label.config(image=white_img)
        predicted_label.config(image=white_img)
        penguin_actual.config(text="N/A")
        penguin_pred.config(text="N/A")

        if next_index < data.shape[1]:
            # Update Penguin Details
            bill_length_amount.config(text=str(current_penguin_data_display[0]))
            bill_depth_amount.config(text=str(current_penguin_data_display[1]))
            flipper_length_amount.config(text=str(current_penguin_data_display[2]))
            body_mass_amount.config(text=str(current_penguin_data_display[3]))

            # Update Button text
            button.config(text="Predict Penguin")
        else:
            # Update Button text
            button.config(text="At Data End")


# Create a window
window = tk.Tk()
window.title("Penguin Identifier")

# Create the images
adelie_img = Image.open("adelie_img.png").resize((1024//3,683//3), Image.LANCZOS)
gentoo_img = Image.open("gentoo_img.png").resize((1000//3,673//3), Image.LANCZOS)
white_img = Image.open("white.png").resize((350,230), Image.LANCZOS)
adelie_img = ImageTk.PhotoImage(adelie_img)
gentoo_img = ImageTk.PhotoImage(gentoo_img)
white_img = ImageTk.PhotoImage(white_img)
real_label = tk.Label(image=white_img)
real_label.config(width=400, height=260)
predicted_label = tk.Label(image=white_img)
predicted_label.config(width=400, height=260)

# Create Pred/Actual Labels
penguin_actual = tk.Label(text="N/A")
penguin_pred = tk.Label(text="N/A")

# Create Penguin Details
initial_penguin_data = current_penguin_data
penguin_title = tk.Label(text="Penguin Details")
actual_title = tk.Label(text="The Penguin")
predicted_title = tk.Label(text="Predicted Penguin")
penguin_title.config(font=("Arial", 20))
actual_title.config(font=("Arial", 20))
predicted_title.config(font=("Arial", 20))
bill_length_label = tk.Label(text="Bill Length (mm)")
bill_depth_label = tk.Label(text="Bill Depth (mm)")
flipper_length_label = tk.Label(text="Flipper Length (mm)")
body_mass_label = tk.Label(text="Body Mass (mm)")
bill_length_label.config(font=("Arial", 12, "bold"))
bill_depth_label.config(font=("Arial", 12, "bold"))
flipper_length_label.config(font=("Arial", 12, "bold"))
body_mass_label.config(font=("Arial", 12, "bold"))
bill_length_amount = tk.Label(text=str(current_penguin_data_display[0]))
bill_depth_amount = tk.Label(text=str(current_penguin_data_display[1]))
flipper_length_amount = tk.Label(text=str(current_penguin_data_display[2]))
body_mass_amount = tk.Label(text=str(current_penguin_data_display[3]))

# Create Classifier Label
classifier_label = tk.Label(text=f"Using Classifier: th = {th_for_printing}, th0 = {th0_for_printing}")

# Create Predict Button
button = tk.Button(text="Predict Penguin", command=predict_penguin)

# Place Headers
penguin_title.grid(row = 0, column = 0, pady = 2, columnspan = 2)
actual_title.grid(row = 0, column = 2, pady = 2)
predicted_title.grid(row = 0, column = 3, pady = 2)

# Place Penguin Detail Labels
bill_length_label.grid(row = 1, column = 0, pady = 2, padx = 20)
bill_length_amount.grid(row = 1, column = 1, pady = 2, padx=(0, 50))
bill_depth_label.grid(row = 2, column = 0, pady = 2, padx = 20)
bill_depth_amount.grid(row = 2, column = 1, pady = 2, padx=(0, 50))
flipper_length_label.grid(row = 3, column = 0, pady = 2, padx = 20)
flipper_length_amount.grid(row = 3, column = 1, pady = 2, padx=(0, 50))
body_mass_label.grid(row = 4, column = 0, pady = 2, padx = 20)
body_mass_amount.grid(row = 4, column = 1, pady = 2, padx=(0, 50))

# Place Images
real_label.grid(row = 1, column = 2, rowspan = 3, padx = 5, pady = 5)
predicted_label.grid(row = 1, column = 3, rowspan = 3, padx = 5, pady = 5)
penguin_actual.grid(row = 4, column = 2, rowspan = 1, padx = 5, pady = 5)
penguin_pred.grid(row = 4, column = 3, rowspan = 1, padx = 5, pady = 5)

# Place Classifier
classifier_label.grid(row = 5, columnspan = 5, pady = 2)

# Place Button
button.grid(row = 6, columnspan = 5, pady = 2)

# Start the main loop
window.mainloop()