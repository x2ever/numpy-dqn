import sys
import time
import io
import cv2
import numpy as np
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from threading import Thread
import time
import matplotlib.pyplot as plt

def get_img_from_fig(fig, dpi=180):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=180)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img

class LogFileEventHandler(FileSystemEventHandler):

    def __init__(self):
        super().__init__()
        self.show = True
        self.show_graph = Thread(target=self.process, args=())
        self.show_graph.start()
        self.count = 1

    def on_modified(self, event):
        self.show = False
        self.show_graph.join()
        self.show_graph = Thread(target=self.process, args=())
        self.count += 1
        self.show = True
        self.show_graph.start()

    def process(self):
        with open("./logs\\log.txt", 'r') as f:
            x = list()
            y = list()
            y2 = list()
            while True:
                data = f.readline()
                if not data:
                    break
                temp = data.replace("\n", '').split("\t")
                x.append(int(temp[0]))
                y.append(float(temp[1]))
                y2.append(float(temp[2]))

        fig, axs = plt.subplots(2, 1, constrained_layout=True, figsize=(6, 9))
        axs[0].plot(x, y)
        axs[0].set_title('Episode Reward')
        axs[0].set_xlabel("Epoch")
        axs[0].set_ylabel("Reward")
        
        axs[1].plot(x, y2)
        axs[1].set_title('Deep Q Network Cost')
        axs[1].set_xlabel("Epoch")
        axs[1].set_ylabel("Cost")

        fig.suptitle("Training Logs\n", fontsize=28)

        plot_img_np = get_img_from_fig(fig)
        plt.close()
    
        while self.show:
            cv2.imshow("Training Logs", plot_img_np)
            cv2.waitKey(10)
        cv2.destroyAllWindows()

            

class LogWatcher:
    def __init__(self, src_path):
        self.__src_path = src_path
        self.__event_observer = Observer()
        self.__event_handler = LogFileEventHandler()

    def run(self):
        self.start()
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            self.stop()
            self.__event_handler.show = False

    def start(self):
        self.__schedule()
        self.__event_observer.start()

    def stop(self):
        self.__event_observer.stop()
        self.__event_observer.join()

    def __schedule(self):
        self.__event_observer.schedule(
            self.__event_handler,
            self.__src_path,
            recursive=True
        )

if __name__ == "__main__":
    LogWatcher("./logs").run()