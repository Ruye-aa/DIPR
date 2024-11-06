import time

class Timer:
    def __init__(self):
        self.total_prepare_time = 0
        self.total_process_time = 0
        self.total_pose_time = 0
        self.count_prepare_time = 0
        self.count_process_time = 0
        self.count_pose_time = 0
        self.last_time = time.time()

    def reset(self):
        self.total_prepare_time = 0
        self.total_process_time = 0
        self.total_pose_time = 0
        self.count_prepare_time = 0
        self.count_process_time = 0
        self.count_pose_time = 0
        
        self.last_time = time.time()

    def record_time(self):
        self.last_time = time.time()

    def add_prepare_time(self):
        current_time = time.time()
        self.total_prepare_time += current_time - self.last_time
        self.count_prepare_time += 1
        self.last_time = current_time

    def add_process_time(self):
        current_time = time.time()
        self.total_process_time += current_time - self.last_time
        self.count_process_time += 1
        self.last_time = current_time

    def add_pose_time(self):
        current_time = time.time()
        self.total_pose_time += current_time - self.last_time
        self.count_pose_time += 1
        self.last_time = current_time

    def get_prepare_time(self):
        return self.total_prepare_time / (self.count_prepare_time + 1e-12)

    def get_process_time(self):
        return self.total_process_time / (self.count_process_time + 1e-12)

    def get_pose_time(self):
        return self.total_pose_time / (self.count_pose_time + 1e-12)

    def tostring(self):
        summary = 'time: '
        if self.count_prepare_time > 0:
            summary += '{:.3f}s/'.format(self.get_prepare_time())
        if self.count_pose_time > 0:
            summary += '{:.3f}s/'.format(self.get_pose_time())
        summary += '{:.3f}s'.format(self.get_process_time())
        return summary
