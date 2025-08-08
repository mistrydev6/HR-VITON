import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import torch

from preprocessing import get_dataset
from inference import inference


def tensor_to_image(tensor):
    """Convert a [-1,1] tensor image to PIL.Image"""
    tensor = tensor.clone().detach().cpu()
    tensor = (tensor + 1) / 2  # [-1,1] -> [0,1]
    tensor = tensor.squeeze(0)
    array = tensor.permute(1, 2, 0).numpy()
    array = (array * 255).clip(0, 255).astype("uint8")
    return Image.fromarray(array)


class HRVitonGUI:
    def __init__(self, master):
        self.master = master
        master.title("HR-VITON Demo")

        self.person_path = tk.StringVar()
        self.cloth_path = tk.StringVar()
        self.mask_path = tk.StringVar()

        tk.Label(master, text="Person image:").grid(row=0, column=0, sticky="e")
        tk.Entry(master, textvariable=self.person_path, width=50).grid(row=0, column=1)
        tk.Button(master, text="Browse", command=self.browse_person).grid(row=0, column=2)

        tk.Label(master, text="Cloth image:").grid(row=1, column=0, sticky="e")
        tk.Entry(master, textvariable=self.cloth_path, width=50).grid(row=1, column=1)
        tk.Button(master, text="Browse", command=self.browse_cloth).grid(row=1, column=2)

        tk.Label(master, text="Cloth mask:").grid(row=2, column=0, sticky="e")
        tk.Entry(master, textvariable=self.mask_path, width=50).grid(row=2, column=1)
        tk.Button(master, text="Browse", command=self.browse_mask).grid(row=2, column=2)

        tk.Button(master, text="Run", command=self.run).grid(row=3, column=1)

        self.result_panel = tk.Label(master)
        self.result_panel.grid(row=4, column=0, columnspan=3)

    def browse_person(self):
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
        if path:
            self.person_path.set(path)

    def browse_cloth(self):
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
        if path:
            self.cloth_path.set(path)

    def browse_mask(self):
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
        if path:
            self.mask_path.set(path)

    def run(self):
        person = self.person_path.get()
        cloth = self.cloth_path.get()
        mask = self.mask_path.get()
        if not (person and cloth and mask):
            messagebox.showerror("Error", "Please select all required files")
            return
        try:
            data = get_dataset(person, cloth, mask)
            with torch.no_grad():
                result, _ = inference(data)
            image = tensor_to_image(result)
            photo = ImageTk.PhotoImage(image)
            self.result_panel.configure(image=photo)
            self.result_panel.image = photo
        except Exception as e:
            messagebox.showerror("Inference failed", str(e))


if __name__ == "__main__":
    root = tk.Tk()
    gui = HRVitonGUI(root)
    root.mainloop()
