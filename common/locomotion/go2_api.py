#!/usr/bin/env python3
"""go2_api: управление Go2 на трассе соревнования

Контроллер участника создаёт Go2 и в цикле читает датчики, выбирает движение
и задаёт скорость через drive(), модель ходьбы управляет ногами и равновесием
Участник пишет навигацию и обработку данных датчиков

    from go2_api import Go2
    bot = Go2()
    while bot.step():
        x, y, yaw = bot.pose()          # Положение по GPS и IMU
        ranges = bot.lidar()            # Расстояния до препятствий с 3D-лидара
        bot.drive(vx=0.4, vyaw=0.0)     # Идём вперёд со скоростью 0,4 м/с

Датчики: pose() возвращает (x, y, yaw), lidar() список расстояний,
image() байты кадра передней камеры, imu() углы (roll, pitch, yaw),
time() время симуляции в секундах
Позы: stand() удерживает стойку, crouch(height) включает приседание
Пример вызова для низкой перекладины: crouch(0.18)
Движение: drive(vx, vyaw, vy=0), vx вперёд в м/с (±0,8),
vyaw поворот в рад/с (±1,5), vy вбок в м/с (±0,4)
Робот движется с полной физикой, обычная скорость ходьбы 0,4-0,5 м/с
Резкие повороты и большая скорость могут привести к падению

Go2(enable_ros2=True) включает узел ROS 2 с публикацией /scan (LaserScan),
/odom (Odometry), /imu, /camera/image_raw и подпиской на /cmd_vel (Twist)
Можно управлять через Nav2 или собственный SLAM, без Python API
Если rclpy не установлен, мост отключается, а Python API продолжает работать
Ходьба, датчики и этот модуль предоставлены организаторами,
участник изменяет свой контроллер
"""
import math
import os
import sys

import struct

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from controller import Robot                       # noqa: E402  # API Webots
from policy_walk import PolicyWalk                 # noqa: E402  # Модель ходьбы MNN из прошивки Go2


class RemoteArm:
    """Клиент стационарного манипулятора на уровнях 2 и 3
    Манипулятор стоит на карте рядом с заданием и получает команды по радио
    Вызовы не ждут выполнения, поэтому робот может двигаться одновременно
    Для обновления состояния нужно продолжать вызывать robot.step()

    Координаты заданы относительно центра основания манипулятора на полу,
    ось z направлена вверх, остальные оси совпадают с осями станции
    Радиус досягаемости около 0,55 м

        robot.arm.move(x, y, z)        переместить захват к точке, наконечник смотрит вниз
        robot.arm.at_target()         достигнута ли последняя цель move(), допуск ±5 см
        robot.arm.tip()                текущая позиция захвата (x, y, z)
        robot.arm.grab() / release()   включить или выключить электромагнит, захват до 9 см
        robot.arm.holding()            удерживается ли объект
        robot.arm.stow()               сложить манипулятор
        robot.arm.request_image() / image()   запросить или получить последний кадр камеры
                                              результат: (w, h, bgra_bytes) или None
        robot.arm.request_depth() / depth()   запросить или получить кадр глубины
                                              результат: (w, h, float32_bytes_m) или None
    """

    def __init__(self, robot, ts):
        self._tx = None
        self._rx = None
        try:
            self._tx = robot.getDevice("arm_tx")
            self._rx = robot.getDevice("arm_rx")
        except Exception:  # noqa: BLE001
            pass
        if self._rx:
            self._rx.enable(ts)
        self.ok = self._tx is not None and self._rx is not None
        self._tip = (0.0, 0.0, 0.0)
        self._holding = False
        self._at_target = False
        self._img = None
        self._dep = None
        self._cool = 0                 # Сколько шагов игнорировать старый at_target после новой команды

    def _poll(self):
        """Прочитать все пакеты телеметрии, вызывается из Go2.step()"""
        if not self.ok:
            return
        if self._cool > 0:
            self._cool -= 1
        while self._rx.getQueueLength() > 0:
            data = self._rx.getBytes()
            self._rx.nextPacket()
            if not data:
                continue
            op = data[:1]
            if op == b'T' and len(data) >= 15:
                x, y, z, hold, at = struct.unpack("<3f2B", data[1:15])
                self._tip, self._holding, self._at_target = (x, y, z), bool(hold), bool(at)
            elif op == b'I' and len(data) >= 5:
                w, h = struct.unpack("<2H", data[1:5])
                self._img = (w, h, data[5:])
            elif op == b'D' and len(data) >= 5:
                w, h = struct.unpack("<2H", data[1:5])
                self._dep = (w, h, data[5:])

    def _send(self, payload):
        if self.ok:
            self._tx.send(payload)

    # Команды без ожидания выполнения
    def move(self, x, y, z):
        self._at_target = False
        self._cool = 30                # Около 0,12 с игнорируем телеметрию, отправленную до получения команды
        self._send(b'M' + struct.pack("<3f", float(x), float(y), float(z)))

    def grab(self):
        self._send(b'G')

    def release(self):
        self._send(b'R')

    def stow(self):
        self._at_target = False
        self._cool = 30
        self._send(b'S')

    def request_image(self):
        self._send(b'I')

    def request_depth(self):
        self._send(b'D')

    # Состояние из последнего пакета телеметрии, обновляется примерно каждые 32 мс
    def at_target(self):
        return self._at_target and self._cool == 0

    def tip(self):
        return self._tip

    def holding(self):
        return self._holding

    def image(self):
        return self._img

    def depth(self):
        return self._dep


class Go2:
    # Пределы скорости в м/с и рад/с, обычная скорость ходьбы: 0,4-0,5 м/с
    # Предел 0,8 м/с не гарантирует устойчивость, движение боком задаётся через vy в drive()
    MAX_VX = 0.8
    MAX_VY = 0.4
    MAX_VYAW = 1.5

    def __init__(self, enable_ros2=False, lidar=True, camera=True):
        self.robot = Robot()
        self.ts = int(self.robot.getBasicTimeStep())
        self.dt = self.ts / 1000.0

        # Модель MNN из прошивки Go2 управляет моментами 12 суставов ног
        # Движение рассчитывается с полной физикой, как в проверенных эталонных симуляциях
        self._walk = PolicyWalk(self.robot)

        # Клиент стационарного манипулятора для уровней 2 и 3, на остальных ok=False
        self.arm = RemoteArm(self.robot, self.ts)

        self._gps = self._dev("gps")
        self._imu = self._dev("imu")
        self._lidar = self._dev("lidar") if lidar else None
        self._camera = self._dev("camera") if camera else None
        for d in (self._gps, self._imu, self._lidar, self._camera):
            if d is not None:
                d.enable(self.ts)

        self._bridge = _Ros2Bridge(self) if enable_ros2 else None

    def _dev(self, name):
        try:
            return self.robot.getDevice(name)
        except Exception:              # noqa: BLE001
            return None

    # Команды
    def drive(self, vx=0.0, vyaw=0.0, vy=0.0):
        """Задать скорость корпуса: vx вперёд в м/с, vyaw поворот в рад/с,
        vy движение вбок в м/с, положительные vy и vyaw направлены влево
        У движения есть инерция и небольшая задержка,
        столкновение со стеной на скорости может привести к падению"""
        self._walk.command(
            max(-self.MAX_VX, min(self.MAX_VX, float(vx))),
            max(-self.MAX_VY, min(self.MAX_VY, float(vy))),
            max(-self.MAX_VYAW, min(self.MAX_VYAW, float(vyaw))),
        )

    def stop(self):
        self.drive(0.0, 0.0)

    def stand(self):
        """Удерживать стойку PD-регулятором, приостановив модель ходьбы"""
        self._walk.stand()

    def crouch(self, height=0.18):
        """Опустить корпус в заданную позу приседания"""
        self._walk.crouch(height)

    # Шаг симуляции: обновляем ходьбу, рассчитываем физику и читаем датчики
    def step(self):
        self._walk.tick()
        r = self.robot.step(self.ts)
        self.arm._poll()                       # Читаем телеметрию манипулятора, если он есть
        if self._bridge is not None:
            self._bridge.publish()             # После шага датчики содержат свежие данные, можно их публиковать
            self._bridge.spin()
        return r != -1

    # Датчики
    def pose(self):
        """Положение (x, y, yaw) в мире: координаты GPS в метрах, курс IMU в радианах"""
        x, y, _ = self._gps.getValues() if self._gps else (0.0, 0.0, 0.0)
        yaw = self._imu.getRollPitchYaw()[2] if self._imu else 0.0
        return (x, y, yaw)

    def position(self):
        return tuple(self._gps.getValues()) if self._gps else (0.0, 0.0, 0.0)

    def imu(self):
        return tuple(self._imu.getRollPitchYaw()) if self._imu else (0.0, 0.0, 0.0)

    def lidar(self):
        """Все слои 3D-лидара для поиска стен и навигации
        Возвращает horizontalResolution × numberOfLayers расстояний в метрах,
        по слоям, inf означает отсутствие отражения
        Параметры lidar_info() позволяют собрать массив [слой][азимут]
        Слои охватывают стены при разной высоте и наклоне корпуса
        Для карты препятствий выбирайте по каждому азимуту ближайшую точку выше пола,
        так напольные контрольные площадки не попадут в препятствия
        Нижние и задние лучи задевают корпус примерно в 0,2 м от датчика,
        эти отражения нужно убрать, пример кода есть в API.md"""
        if not self._lidar:
            return []
        img = self._lidar.getRangeImage()      # До первого robot.step() возвращается None
        return list(img) if img else []

    def lidar_layer(self, layer=None):
        """Один горизонтальный слой лидара, расстояния в метрах
        По умолчанию берётся средний слой на высоте датчика,
        он может пройти над низкой стеной, для поиска стен используйте lidar()
        Параметр layer выбирает слой от 0 до layers-1"""
        if not self._lidar:
            return []
        if layer is None:
            layer = self._lidar.getNumberOfLayers() // 2
        img = self._lidar.getLayerRangeImage(layer)   # До первого robot.step() возвращается None
        return list(img) if img else []

    def lidar_info(self):
        """Параметры лидара: (h_fov_rad, h_res, max_range, layers, v_fov_rad)
        Данные lidar() можно разложить в массив [layer][h_res]
        Угол луча с индексом i примерно равен -h_fov/2 + i*h_fov/(h_res-1)"""
        if not self._lidar:
            return (0.0, 0, 0.0, 0, 0.0)
        return (self._lidar.getFov(), self._lidar.getHorizontalResolution(),
                self._lidar.getMaxRange(), self._lidar.getNumberOfLayers(),
                self._lidar.getVerticalFov())

    def image(self):
        """Байты BGRA передней камеры (width×height×4) или None, размеры даёт camera_info()"""
        return self._camera.getImage() if self._camera else None

    def camera_info(self):
        if not self._camera:
            return (0, 0, 0.0)
        return (self._camera.getWidth(), self._camera.getHeight(), self._camera.getFov())

    def time(self):
        return self.robot.getTime()


class _Ros2Bridge:
    """Мост ROS 2 с публикацией /scan, /odom, /imu и /camera/image_raw
    Команды /cmd_vel передаются в drive(), без rclpy мост отключается"""

    def __init__(self, bot: "Go2"):
        self.bot = bot
        self.ok = False
        try:
            import rclpy
            from rclpy.node import Node
            from geometry_msgs.msg import Twist
            from sensor_msgs.msg import LaserScan, Imu
            from nav_msgs.msg import Odometry
            self._rclpy = rclpy
            self._LaserScan, self._Imu, self._Odometry = LaserScan, Imu, Odometry
            if not rclpy.ok():
                rclpy.init(args=None)
            self.node = rclpy.create_node("go2_participant")
            self.node.create_subscription(Twist, "cmd_vel", self._on_cmd, 10)
            self.pub_scan = self.node.create_publisher(LaserScan, "scan", 10)
            self.pub_odom = self.node.create_publisher(Odometry, "odom", 10)
            self.pub_imu = self.node.create_publisher(Imu, "imu", 10)
            self.ok = True
        except Exception as e:  # noqa: BLE001  # Без ROS 2 Python API продолжает работать
            print(f"[go2_api] ROS2 bridge disabled ({e}); use the Python API.", flush=True)

    def _on_cmd(self, msg):
        self.bot.drive(msg.linear.x, msg.angular.z)

    def spin(self):
        if self.ok:
            self._rclpy.spin_once(self.node, timeout_sec=0.0)

    def publish(self):
        if not self.ok:
            return
        now = self.node.get_clock().now().to_msg()
        # /scan: LaserScan двумерный, поэтому берём средний горизонтальный слой лидара
        ranges = self.bot.lidar_layer()
        if ranges:
            info = self.bot.lidar_info()
            fov, n, mx = info[0], info[1], info[2]
            s = self._LaserScan()
            s.header.stamp = now
            s.header.frame_id = "base_link"
            s.angle_min = -fov / 2.0
            s.angle_max = fov / 2.0
            s.angle_increment = fov / max(1, n - 1)
            s.range_min = 0.05
            s.range_max = float(mx)
            s.ranges = [float(r) for r in ranges]
            self.pub_scan.publish(s)
        # /odom: положение по GPS и IMU, без ковариации
        x, y, yaw = self.bot.pose()
        od = self._Odometry()
        od.header.stamp = now
        od.header.frame_id = "odom"
        od.child_frame_id = "base_link"
        od.pose.pose.position.x, od.pose.pose.position.y = float(x), float(y)
        od.pose.pose.orientation.z = math.sin(yaw / 2.0)
        od.pose.pose.orientation.w = math.cos(yaw / 2.0)
        self.pub_odom.publish(od)
        # /imu: только ориентация
        r, p, yw = self.bot.imu()
        im = self._Imu()
        im.header.stamp = now
        im.header.frame_id = "base_link"
        cy, sy = math.cos(yw / 2), math.sin(yw / 2)
        cp, sp = math.cos(p / 2), math.sin(p / 2)
        cr, sr = math.cos(r / 2), math.sin(r / 2)
        im.orientation.w = cr * cp * cy + sr * sp * sy
        im.orientation.x = sr * cp * cy - cr * sp * sy
        im.orientation.y = cr * sp * cy + sr * cp * sy
        im.orientation.z = cr * cp * sy - sr * sp * cy
        self.pub_imu.publish(im)
