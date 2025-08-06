# GUI_PID_control
Checking and controlling robots state through GUI map which is formed subplot
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-Robot 상태 모니터링 GUI (2×2, 하단은 상태 텍스트)
 - idx1 (좌상): 커스텀 맵 (흑백)
 - idx2 (우상): 카메라 화면 (맵과 동일 크기)
 - idx3 (좌하): 3대 로봇 각각의 속도(선형, 각속도)
 - idx4 (우하): 3대 로봇 각각의 waypoint, 상태
"""

import sys
import math
import threading
import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge

from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout
from PyQt5.QtCore import QTimer
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

NUM_ROBOTS = 3

custom_map = np.array([
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
], dtype=np.uint8)
h, w = custom_map.shape

class RobotMonitor(Node):
    """
    ROS2 Node: Odometry, 상태, 카메라 토픽 구독 후 내부에 저장
    """
    def __init__(self):
        super().__init__('multi_robot_monitor')
        self.lock = threading.Lock()
        self.robot_speed = {}    # {id: (linear, angular)}
        self.robot_state = {}    # {id: (waypoint, status)}
        self.robot_image = np.zeros((h, w, 3), dtype=np.uint8)
        qos = QoSProfile(depth=10)

        self.bridge = CvBridge()
        # Odometry/상태: 3대 로봇용으로 각각 구독
        for i in range(1, NUM_ROBOTS+1):
            self.create_subscription(
                Odometry, f'/odom{i}', self.make_odom_cb(i), qos)
            self.create_subscription(
                String, f'/state{i}', self.make_state_cb(i), qos)
        # 카메라
        self.create_subscription(
            Image, '/monitor_camera/image_raw', self._on_image, qos)

    def make_odom_cb(self, rid):
        def cb(msg):
            lin = float(msg.twist.twist.linear.x)
            ang = float(msg.twist.twist.angular.z)
            with self.lock:
                self.robot_speed[rid] = (lin, ang)
        return cb

    def make_state_cb(self, rid):
        def cb(msg):
            try:
                parts = msg.data.split(',')
                wp = int(parts[1])
                st = parts[2]
                with self.lock:
                    self.robot_state[rid] = (wp, st)
            except Exception as e:
                pass
        return cb

    def _on_image(self, msg):
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            img_small = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_NEAREST)
            with self.lock:
                self.robot_image = img_small
        except Exception as e:
            self.get_logger().error(f"Image CB error: {e}")

class MainWindow(QMainWindow):
    """
    2×2 subplot: (좌상)맵, (우상)카메라, (좌하)속도텍스트, (우하)상태텍스트
    """
    def __init__(self, node: RobotMonitor):
        super().__init__()
        self.node = node
        self.setWindowTitle('Multi-Robot Monitor')
        self.resize(900, 900)

        fig = Figure()
        self.canvas = FigureCanvas(fig)
        axes = np.array(fig.subplots(2, 2))
        self.ax_map,  self.ax_cam = axes[0, 0], axes[0, 1]
        self.ax_stat1, self.ax_stat2 = axes[1, 0], axes[1, 1]
        for ax in [self.ax_map, self.ax_cam, self.ax_stat1, self.ax_stat2]:
            ax.set_box_aspect(1)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.addWidget(self.canvas)
        self.setCentralWidget(container)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.spin_and_update)
        self.timer.start(100)  # 10Hz

    def spin_and_update(self):
        rclpy.spin_once(self.node, timeout_sec=0.001)
        self.update_plot()

    def update_plot(self):
        with self.node.lock:
            speeds = dict(self.node.robot_speed)
            states = dict(self.node.robot_state)
            cam_img = self.node.robot_image.copy()

        # 1) Custom Map (좌상)
        self.ax_map.clear()
        img_map = (1 - custom_map) * 255
        self.ax_map.imshow(
            img_map, cmap='gray', origin='lower',
            extent=[0, w, 0, h], interpolation='nearest')
        self.ax_map.set_title('Custom Map')
        self.ax_map.axis('off')

        # 2) Camera View (우상)
        self.ax_cam.clear()
        self.ax_cam.imshow(
            cam_img, origin='lower',
            extent=[0, w, 0, h], interpolation='nearest')
        self.ax_cam.set_title('Camera View')
        self.ax_cam.axis('off')

        # 3) (좌하) 3대 로봇 속도 텍스트
        self.ax_stat1.clear()
        self.ax_stat1.axis('off')
        self.ax_stat1.set_title('Speed Monitoring')
        for idx, rid in enumerate(range(1, NUM_ROBOTS+1), start=0):
            lin, ang = speeds.get(rid, (0.0, 0.0))
            text = (f'[Robot {rid}]\n'
                    f'  Linear : {lin:.2f} m/s\n'
                    f'  Angular: {ang:.2f} rad/s')
            self.ax_stat1.text(
                0.05, 1 - idx*0.32, text, va='top', ha='left', fontsize=13, transform=self.ax_stat1.transAxes)

        # 4) (우하) 3대 로봇 waypoint/상태 텍스트
        self.ax_stat2.clear()
        self.ax_stat2.axis('off')
        self.ax_stat2.set_title('Task State')
        for idx, rid in enumerate(range(1, NUM_ROBOTS+1), start=0):
            wp, st = states.get(rid, ('-', '-'))
            text = (f'[Robot {rid}]\n'
                    f'  Waypoint: {wp}\n'
                    f'  State   : {st}')
            self.ax_stat2.text(
                0.05, 1 - idx*0.32, text, va='top', ha='left', fontsize=13, transform=self.ax_stat2.transAxes)

        self.canvas.draw_idle()

def main():
    rclpy.init()
    node = RobotMonitor()
    app = QApplication(sys.argv)
    win = MainWindow(node)
    win.show()
    ret = app.exec_()
    node.destroy_node()
    rclpy.shutdown()
    sys.exit(ret)

if __name__ == '__main__':
    main()
