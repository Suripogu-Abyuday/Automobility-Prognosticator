import tkinter as tk
from tkinter import filedialog, messagebox
import os
import subprocess
import new
# Define Colors and Fonts for Styling
PRIMARY_BG = "#2d3436"
SECONDARY_BG = "#1e272e"
BUTTON_COLOR = "#0984e3"
TEXT_COLOR = "#ffffff"
HIGHLIGHT_COLOR = "#d63031"
FONT_LARGE = ("Helvetica", 18, "bold")
FONT_MEDIUM = ("Helvetica", 14)
FONT_SMALL = ("Helvetica", 10)

def browse_file(entry_field, file_type):
    """Open a file dialog to select a file and set its path in the entry field"""
    file_path = filedialog.askopenfilename(title=f"Select {file_type} File")
    if file_path:
        entry_field.delete(0, tk.END)
        entry_field.insert(0, file_path)

def browse_folder(entry_field):
    """Open a file dialog to select a folder and set its path in the entry field"""
    folder_path = filedialog.askdirectory(title="Select Output Folder")
    if folder_path:
        entry_field.delete(0, tk.END)
        entry_field.insert(0, folder_path)

def run_tracking():
    """Run the vehicle tracking system with the provided inputs"""
    video_path = video_entry.get()
    model_path = model_entry.get()
    output_path = output_entry.get()

    if not video_path or not os.path.exists(video_path):
        messagebox.showerror("Error", "Please provide a valid video file.")
        return

    if not model_path or not os.path.exists(model_path):
        messagebox.showerror("Error", "Please provide a valid model file.")
        return

    if not output_path or not os.path.isdir(output_path):
        messagebox.showerror("Error", "Please provide a valid output directory.")
        return

    # Run the new.py script with the provided arguments
    try:
        # subprocess.run([
        #     'python', 'new.py',
        #     '--video', video_path,
        #     '--model', model_path,
        #     '--output', output_path
        # ], check=True)
        new.main(video_path, model_path, output_path)
        messagebox.showinfo("Success", f"Processing complete. Results saved in {output_path}")
    except subprocess.CalledProcessError as e:
        messagebox.showerror("Error", f"An error occurred: {e}")

# Create the main application window
app = tk.Tk()
app.title("\ud83d\ude97 Vehicle Tracking System")
app.geometry("800x500")
app.configure(bg=PRIMARY_BG)

# Header Section
header_frame = tk.Frame(app, bg=SECONDARY_BG, pady=10, padx=10)
header_frame.pack(fill="x", pady=10)
header_label = tk.Label(header_frame, text="\ud83d\ude97 Vehicle Tracking System \ud83d\ude97", font=FONT_LARGE, bg=SECONDARY_BG, fg=TEXT_COLOR)
header_label.pack()

# Input Section
input_frame = tk.Frame(app, bg=PRIMARY_BG)
input_frame.pack(fill="x", pady=10, padx=20)

def create_input_row(label_text, command, entry_field, file_type=None):
    row_frame = tk.Frame(input_frame, bg=PRIMARY_BG, pady=10)
    row_frame.pack(fill="x", pady=5)

    label = tk.Label(row_frame, text=label_text, font=FONT_MEDIUM, bg=PRIMARY_BG, fg=TEXT_COLOR)
    label.pack(side="left", padx=(0, 10))

    entry = tk.Entry(row_frame, font=FONT_MEDIUM, width=40, bg=SECONDARY_BG, fg=TEXT_COLOR, insertbackground=TEXT_COLOR)
    entry.pack(side="left", fill="x", expand=True, padx=(0, 10))

    button = tk.Button(row_frame, text="Browse", font=FONT_SMALL, command=lambda: command(entry, file_type) if file_type else command(entry), bg=BUTTON_COLOR, fg=TEXT_COLOR, activebackground=HIGHLIGHT_COLOR, activeforeground=TEXT_COLOR)
    button.pack(side="right")

    return entry

# Create Input Rows
video_entry = create_input_row("Video File:", browse_file, None, "Video")
model_entry = create_input_row("Model File:", browse_file, None, "Model")
output_entry = create_input_row("Output Folder:", browse_folder, None)

# Run Button Section
button_frame = tk.Frame(app, bg=PRIMARY_BG)
button_frame.pack(pady=20)
run_button = tk.Button(button_frame, text="\ud83d\ude80 Run Tracking", font=("Helvetica", 16, "bold"), command=run_tracking, bg=HIGHLIGHT_COLOR, fg=TEXT_COLOR, activebackground=BUTTON_COLOR, activeforeground=TEXT_COLOR, width=20)
run_button.pack()

# Footer Section
footer_frame = tk.Frame(app, bg=SECONDARY_BG, pady=10)
footer_frame.pack(side="bottom", fill="x")
footer_label = tk.Label(footer_frame, text="\u00a9 2024 Rugvidh Solutions - All Rights Reserved", font=FONT_SMALL, bg=SECONDARY_BG, fg=TEXT_COLOR)
footer_label.pack()

# Run the application
app.mainloop()