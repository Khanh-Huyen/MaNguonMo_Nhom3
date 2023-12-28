import tkinter as tk
import cv2
import numpy as np
from PIL import Image, ImageTk
import time as t
from tkinter import ttk, font,filedialog, Label
from current import process, sudoku
from models import model_wrapper
from preprocessing import preprocess
import random
from itertools import product
import tkinter.simpledialog as simpledialog
# Các biến toàn cục
video_path = None
processing = False
frame_rate = 30
selected = None
invalid_selected = False

# Cài đặt Giao diện người giải Sudoku
root = tk.Tk()
root.geometry("700x700")
root.title("Trò chơi sudoku")
root.iconbitmap(r"C:\Users\hoang ty\Downloads\sudoku2.ico")
root.config(background="white")

# Giải sudoku
def select_video():
    global video_path
    video_path = filedialog.askopenfilename()

def start_processing():
    global processing, frame_rate
    processing = True
    cap = cv2.VideoCapture(video_path)

    frame_width = 960
    frame_height = 720
    cap.set(3, frame_width)
    cap.set(4, frame_height)
    cap.set(10, 150)

    my_model = model_wrapper.model_wrapper(None, False, None, "model.hdf5")

    prev = 0
    seen = dict()

    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    # Tính toán vị trí x, y để canvas hiển thị ở giữa dưới giao diện
    canvas_x = (screen_width - frame_width) / 11
    canvas_y = (screen_height - frame_height) / 2

    canvas = tk.Canvas(root, width=frame_width, height=frame_height)
    canvas.place(x=canvas_x, y=canvas_y)

    while processing:
        time_elapsed = t.time() - prev
        success, img = cap.read()

        if time_elapsed > 1. / frame_rate:
            prev = t.time()
            img_result = img.copy()
            img_corners = img.copy()

            processed_img = preprocess.preprocess(img)
            corners = process.find_contours(processed_img, img_corners)

            if corners:
                warped, matrix = process.warp_image(corners, img)
                warped_processed = preprocess.preprocess(warped)
                vertical_lines, horizontal_lines = process.get_grid_lines(warped_processed)
                mask = process.create_grid_mask(vertical_lines, horizontal_lines)
                numbers = cv2.bitwise_and(warped_processed, mask)
                squares = process.split_into_squares(numbers)
                squares_processed = process.clean_squares(squares)
                squares_guesses = process.recognize_digits(squares_processed, my_model)

                if squares_guesses in seen and seen[squares_guesses] is False:
                    continue

                if squares_guesses in seen:
                    process.draw_digits_on_warped(warped, seen[squares_guesses][0], squares_processed)
                    img_result = process.unwarp_image(warped, img_result, corners, seen[squares_guesses][1])
                else:
                    solved_puzzle, time = sudoku.solve_wrapper(squares_guesses)
                    if solved_puzzle is not None:
                        process.draw_digits_on_warped(warped, solved_puzzle, squares_processed)
                        img_result = process.unwarp_image(warped, img_result, corners, time)
                        seen[squares_guesses] = [solved_puzzle, time]
                    else:
                        seen[squares_guesses] = False
        desired_width = 600  # Thay đổi kích thước mong muốn theo nhu cầu
        desired_height = 600

        # Chuyển đổi ảnh sang định dạng RGB để hiển thị trong Tkinter
        img_rgb = cv2.cvtColor(img_result, cv2.COLOR_BGR2RGB)
        img_rgb_resized = cv2.resize(img_rgb, (desired_width, desired_height))
        img_tk = ImageTk.PhotoImage(Image.fromarray(img_rgb_resized))

        # Cập nhật canvas với ảnh mới
        canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
        root.update()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    root.destroy()
    cap.release()
    cv2.destroyAllWindows()
def stop_processing():
    global processing
    processing = False

# Hàm quay lại
def go_back():
    if history_stack:
        # Lấy và gọi hàm trạng thái trước đó từ stack
        previous_state = history_stack.pop()
        previous_state()

# Tạo sudoku
class SudokuGame:
    def __init__(self, size):
        self.size = size
        self.empty_sudoku = self.create_empty_sudoku()
        self.solved_sudoku = self.solve_sudoku()

    def create_empty_sudoku(self):
        R, C = self.size
        N = R * C
        return [[0 for _ in range(N)] for _ in range(N)]

    def solve_sudoku(self):
        return next(sudoku.solve_sudoku(self.size, self.empty_sudoku))

    def remove_numbers(self, cells_to_remove):
        puzzle = [row.copy() for row in self.solved_sudoku]
        all_cells = list(product(range(len(puzzle)), range(len(puzzle[0]))))
        random.shuffle(all_cells)

        for cell in all_cells[:cells_to_remove]:
            puzzle[cell[0]][cell[1]] = 0

        return puzzle

class SudokuGUI:
    def __init__(self, root, sudoku_game):
        self.root = root
        self.sudoku_game = sudoku_game
        self.cell_size = 60
        self.margin = 10

        # Tạo Canvas để vẽ Sudoku
        self.canvas = tk.Canvas(root, width=(len(sudoku_game.empty_sudoku[0])+0.3) * self.cell_size,
                                height=(len(sudoku_game.empty_sudoku) + 0.3) * self.cell_size)
        self.canvas.pack(padx=20,pady=40)

        # Vẽ Sudoku lên Canvas
        self.draw_sudoku()

        # Tạo nút "New Game"
        self.new_game_button = tk.Button(root, text="New Game", command=self.new_game)
        self.new_game_button.pack(side=tk.LEFT, padx=10)

        self.solve_button = tk.Button(root, text="Solve", command=self.solve_sudoku)
        self.solve_button.pack(side=tk.LEFT, padx=10)

        self.btn_stop = tk.Button(root, text="Back", command=go_back)
        self.btn_stop.pack(side=tk.LEFT, padx=10)
    def draw_sudoku(self):
        for i, row in enumerate(self.sudoku_game.empty_sudoku):
            for j, num in enumerate(row):
                x0 = j * self.cell_size + self.margin
                y0 = i * self.cell_size + self.margin
                x1 = x0 + self.cell_size
                y1 = y0 + self.cell_size

                self.canvas.create_rectangle(x0, y0, x1, y1, outline="black", width=2)

                if num != 0:
                    self.canvas.create_text(x0 + self.cell_size / 2, y0 + self.cell_size / 2, text=str(num),
                                            font=("Helvetica", 12, "bold"))

    def new_game(self):
        difficulty_level = random.randint(50, 80)
        total_cells = len(self.sudoku_game.empty_sudoku) * len(self.sudoku_game.empty_sudoku[0])
        cells_to_remove = total_cells * difficulty_level // 100
        # Tạo một trò chơi Sudoku mới
        self.sudoku_game.empty_sudoku = self.sudoku_game.remove_numbers(cells_to_remove)
        # Xóa nội dung cũ trên Canvas
        self.canvas.delete("all")
        # Vẽ trò chơi Sudoku mới
        self.draw_sudoku()
    def solve_sudoku(self):
        # Get the solved Sudoku solution
        solved_sudoku_solution = self.sudoku_game.solved_sudoku
        # Update the GUI with the solved solution
        self.sudoku_game.empty_sudoku = solved_sudoku_solution
        self.canvas.delete("all")
        self.draw_sudoku()

# Hướng dẫn chơi sudoku
history_stack = []


class VideoPlayerApp:
    def __init__(self, master, video_source=0, width=0, height=0):
        self.master = master
        self.master.title("Video Player App")

        self.video_source = video_source
        self.vid = cv2.VideoCapture(self.video_source)
        self.paused = False  # Flag to track whether the video is paused

        self.canvas = tk.Canvas(master, width=width, height=height)
        self.canvas.pack()

        self.btn_play = ttk.Button(master, text="Play", command=self.play)
        self.btn_play.pack(side=tk.LEFT)

        self.btn_pause = ttk.Button(master, text="Pause", command=self.pause)
        self.btn_pause.pack(side=tk.LEFT)

        self.btn_replay = ttk.Button(master, text="Phát lại", command=self.replay)
        self.btn_replay.pack(side=tk.LEFT)

        self.btn_stop = ttk.Button(master, text="back", command=go_back)
        self.btn_stop.pack(side=tk.LEFT)

        self.width = width
        self.height = height
        self.update()

    def play(self):
        self.paused = False
        self.btn_play["state"] = "disabled"
        self.btn_pause["state"] = "enabled"
        self.btn_stop["state"] = "enabled"
        self.btn_replay["state"] = "enabled"

    def pause(self):
        self.paused = True
        self.btn_play["state"] = "enabled"
        self.btn_pause["state"] = "disabled"
        self.btn_stop["state"] = "enabled"
        self.btn_replay["state"] = "enabled"

    def replay(self):
        self.vid.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Đặt vị trí video về đầu
        self.play()

    def update(self):
        if not self.paused:
            ret, frame = self.vid.read()
            if ret:
                frame = cv2.resize(frame, (self.width, self.height))
                self.photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
                self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
        self.master.after(10, self.update)


# Hàm chức năng
def chucnang1():
    history_stack.append(sudoku_gamee)
    [widget.destroy() for widget in root.winfo_children()]
    app = VideoPlayerApp(root, video_source=r"C:\Users\hoang ty\Downloads\sudoku1.mp4", width=600, height=600)


def chucnang2():
    history_stack.append(sudoku_gamee)
    [widget.destroy() for widget in root.winfo_children()]

    b_button = tk.Button(root, text="Quay lại", bg='red', fg='white', command=go_back)
    b_button.pack(side=tk.LEFT, anchor=tk.NW, padx=30, pady=10)

    select_button = tk.Button(root, text="Chọn Video", bg='red', fg='white', command=select_video)
    select_button.pack(side=tk.LEFT, anchor=tk.NW, padx=10, pady=10)

    start_button = tk.Button(root, text="Bắt đầu Xử lý", bg='green', fg='white', command=start_processing)
    start_button.pack(side=tk.LEFT, anchor=tk.NW, padx=30, pady=10)

    stop_button = tk.Button(root, text="Dừng Xử lý", bg='blue', fg='white', command=stop_processing)
    stop_button.pack(side=tk.LEFT, anchor=tk.NW, padx=30, pady=10)


def chucnang3():
    history_stack.append(sudoku_gamee)
    [widget.destroy() for widget in root.winfo_children()]
    size = (3, 3)
    sudoku_game = SudokuGame(size)
    difficulty_level = random.randint(1,80)
    total_cells = len(sudoku_game.empty_sudoku) * len(sudoku_game.empty_sudoku[0])
    cells_to_remove = total_cells * difficulty_level // 100
    sudoku_game.empty_sudoku = sudoku_game.remove_numbers(cells_to_remove)
    gui = SudokuGUI(root, sudoku_game)


def sudoku_gamee():
    lbl = Label(root, text='Chọn chức năng', font='ARIAN 40 bold', fg='red', compound='bottom')
    lbl.pack(pady=40)

    cn1 = tk.Button(root, text="Hướng dẫn chơi", font='ARIAN 40 bold', bg='gray', fg='white', command=chucnang1,
                    width=15)
    cn1.pack(pady=(20, 20))

    cn2 = tk.Button(root, text="Giải Sudoku", font='ARIAN 40 bold', bg='gray', fg='white', command=chucnang2,
                    width=15)
    cn2.pack(pady=(20, 20))

    cn3 = tk.Button(root, text="Tạo Sudoku", font='ARIAN 40 bold', bg='gray', fg='white', command=chucnang3,
                    width=15)
    cn3.pack(pady=(20, 20))


sudoku_gamee()

# Bắt đầu vòng lặp sự kiện Tkinter
root.mainloop()
