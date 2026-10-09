#!/usr/bin/env python3
"""Судья соревнования для трёх уровней, работает через Webots Supervisor

Запускается отдельным контроллером в своём узле Robot с supervisor TRUE
Робот участника не получает доступ к Supervisor API и не может перемещать себя
или начислять баллы, судья читает его положение и проверяет условия в мире

Путь к JSON уровня берётся из controllerArgs[0] или REFEREE_CONFIG,
по умолчанию используется ../config/level.json
Судья засчитывает контрольные точки, проверяет задержку внутри финишного кольца,
завершает попытку при падении, перевороте или истечении времени
Зачтённые точки становятся зелёными, результат записывается в result.json
По настройкам можно записать видео, прохождение оценивается по времени симуляции

Поля result.json: level, reason (finish|fall|timeout), score, base_score, bonus,
checkpoints [{name, scored, sim_time}], sim_time, wall_time, penalties
"""
import json
import math
import os
import sys
import time

from controller import Supervisor

GREEN = [0.1, 0.9, 0.2]
YELLOW = [0.95, 0.85, 0.1]


def _load_config():
    path = None
    if len(sys.argv) > 1:
        path = sys.argv[1]
    path = os.environ.get("REFEREE_CONFIG", path)
    if not path:
        here = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(here, "..", "config", "level.json")
    with open(path) as f:
        return json.load(f), path


class Referee:
    def __init__(self):
        self.sup = Supervisor()
        self.ts = int(self.sup.getBasicTimeStep())
        self.cfg, self.cfg_path = _load_config()

        self.robot = self.sup.getFromDef(self.cfg.get("robot_def", "ROBOT"))
        if self.robot is None:
            print("[referee] FATAL: robot DEF not found", flush=True)
        # У Go2Base узел ROBOT служит неподвижной опорой, а корпус движется через суставы
        # Корпус находится внутри PROTO под DEF BODY по умолчанию
        # Читаем его положение и ориентацию в мире через getPosition() и getOrientation()
        # Для обычного робота читаем сам узел ROBOT, body_def="" принудительно выбирает его
        self._pose_node = None
        if self.robot is not None:
            body_def = self.cfg.get("body_def", "BODY")
            if body_def:
                try:
                    self._pose_node = self.robot.getFromProtoDef(body_def)
                except Exception:  # noqa: BLE001  # Старая версия API или узел без PROTO
                    self._pose_node = None
            if self._pose_node is None:
                self._pose_node = self.robot
        print(f"[referee] pose source: {'BODY' if self._pose_node is not self.robot else 'ROBOT node'}",
              flush=True)

        self.cps = [dict(c, scored=False, sim_time=None) for c in self.cfg["checkpoints"]]
        # В режиме sequential точки проходят по порядку, финиш доступен после последней
        # В режиме any точки можно собирать в любом порядке, они не задают маршрут
        # Финиш в этом режиме доступен сразу
        # Бонус за полное прохождение всё равно требует собрать все точки
        self.order = self.cfg.get("checkpoint_order", "sequential")
        self.finish = self.cfg["finish"]
        self.bonus_pct = self.cfg.get("scoring", {}).get("completion_bonus_pct", 30)
        self.fall = self.cfg.get("fall", {"z_min": 0.12, "tilt_max_deg": 60})
        self.tmo = self.cfg.get("timeouts", {"sim_time_s": 240, "wall_clock_s": 900})
        # REFEREE_RESULT и REFEREE_VIDEO переопределяют пути вывода из настроек
        # Их задаёт автоматическая проверка, без них берём пути из конфигурации
        self.result_path = os.environ.get("REFEREE_RESULT") or self.cfg.get("result_path", "result.json")
        self.video_path = os.environ.get("REFEREE_VIDEO") or self.cfg.get("video_path", "run.mp4")

        self._idx = 0                      # Следующая точка для зачёта в режиме sequential
        self._finish_hold = 0.0            # Время внутри финишного кольца в секундах
        self._penalties = []
        self._wall_start = time.time()
        self._done = False

        # Задания с объектами: каждый нужно оставить в своей зоне, баллы начисляются один раз
        # Когда все объекты расставлены, дверь опускается под пол и открывает короткий путь
        # Настройки: "manipulation": { "tasks": [{name, object_def, x, y, radius, z_min, z_max,
        # points}...], "door_def": "DOOR", "door_drop": 0.6 }
        man = self.cfg.get("manipulation", {})
        self.man_tasks = [dict(t, done=False, sim_time=None) for t in man.get("tasks", [])]
        self._man_nodes = {t["object_def"]: self.sup.getFromDef(t["object_def"])
                           for t in self.man_tasks}
        self._door = self.sup.getFromDef(man.get("door_def", "")) if man.get("door_def") else None
        self._door_drop = float(man.get("door_drop", 0.6))
        self._door_z = None                # Состояние анимации открытия двери
        if self.man_tasks:
            missing = [d for d, n in self._man_nodes.items() if n is None]
            if missing or (man.get("door_def") and self._door is None):
                print(f"[referee] WARNING: manipulation DEFs missing: {missing} "
                      f"door={self._door is not None}", flush=True)

        self._recording = False
        # Записываем видео по настройкам или при REFEREE_RECORD=1
        want_video = self.cfg.get("record_video") or os.environ.get("REFEREE_RECORD") == "1"
        if want_video:
            fn = self.video_path
            try:
                self.sup.movieStartRecording(fn, width=1280, height=720, quality=70,
                                             codec=0, acceleration=1, caption=False)
                self._recording = True
            except Exception as e:  # noqa: BLE001
                print(f"[referee] video disabled: {e}", flush=True)
        print(f"[referee] level {self.cfg.get('level')} — {len(self.cps)} checkpoints, "
              f"config={os.path.basename(self.cfg_path)}", flush=True)

    # Положение и ориентация
    def _pose(self):
        p = self._pose_node.getPosition()          # Координаты движущегося корпуса [x, y, z] в мире
        o = self._pose_node.getOrientation()       # Матрица 3×3, записанная по строкам
        up_z = o[8]                                 # Проекция вертикальной оси корпуса на ось z мира
        yaw = math.atan2(o[3], o[0])
        return p[0], p[1], p[2], up_z, yaw

    def _recolor(self, cp, color):
        node = self.sup.getFromDef(cp.get("def", ""))
        if node is not None:
            f = node.getField("color")
            if f is not None:
                f.setSFColor(color)

    # Подсчёт баллов
    def _base_score(self):
        return (sum(c["points"] for c in self.cps if c["scored"])
                + sum(t["points"] for t in self.man_tasks if t["done"]))

    def _final(self, reason):
        # Баллы начисляем за конечное положение объекта в его зоне
        # Пронос объекта через зону в захвате не считается
        # Один объект занимает одну зону
        for task in self.man_tasks:
            node = self._man_nodes.get(task["object_def"])
            if node is not None:
                px, py, pz = node.getPosition()
                task["done"] = (math.hypot(px - task["x"], py - task["y"]) <= task["radius"]
                                and task.get("z_min", -1e9) <= pz <= task.get("z_max", 1e9))
            else:
                task["done"] = False
        base = self._base_score()
        completed = (all(c["scored"] for c in self.cps)
                     and all(t["done"] for t in self.man_tasks)
                     and reason == "finish")
        bonus = round(base * self.bonus_pct / 100.0) if completed else 0
        score = base + bonus - sum(p.get("cost", 0) for p in self._penalties)
        result = {
            "level": self.cfg.get("level"),
            "reason": reason,
            "score": score,
            "base_score": base,
            "bonus": bonus,
            "completed": completed,
            "checkpoints": [{"name": c["name"], "scored": c["scored"], "sim_time": c["sim_time"]}
                            for c in self.cps],
            "sim_time": round(self.sup.getTime(), 3),
            "wall_time": round(time.time() - self._wall_start, 3),
            "penalties": self._penalties,
            "manipulation": [{"name": t["name"], "done": t["done"], "sim_time": t["sim_time"]}
                             for t in self.man_tasks],
        }
        try:
            with open(self.result_path, "w") as f:
                json.dump(result, f, indent=2)
        except Exception as e:  # noqa: BLE001
            print(f"[referee] result write failed: {e}", flush=True)
        print(f"[referee] DONE reason={reason} score={score} (base {base} + bonus {bonus}) "
              f"sim={result['sim_time']}s", flush=True)
        if self._recording:
            self.sup.movieStopRecording()
            # Ждём окончания кодирования видео, иначе Webots завершит кодировщик и файл потеряется
            # Запись 720p длительностью больше двух минут может кодироваться несколько минут
            # Ограничиваем ожидание 900 секундами на случай зависания
            t0 = time.time()
            while not self.sup.movieIsReady() and time.time() - t0 < 900:
                self.sup.step(self.ts)
            print(f"[referee] video finalised (encode {time.time() - t0:.0f}s, "
                  f"ready={self.sup.movieIsReady()})", flush=True)
        self._done = True
        self.sup.simulationSetMode(Supervisor.SIMULATION_MODE_PAUSE)
        self.sup.simulationQuit(0)

    # Основной цикл
    def run(self):
        if self.robot is None:
            self._final("error")
            return
        # REFEREE_SCREENSHOT задаёт путь к обзорному снимку для примеров и документации
        # Даём сцене отрисоваться, сохраняем снимок с камеры Viewpoint и завершаем запуск без подсчёта
        # При оценке решений эта переменная не задаётся
        shot = os.environ.get("REFEREE_SCREENSHOT")
        if shot:
            for _ in range(40):
                self.sup.step(self.ts)
            self.sup.exportImage(shot, 100)
            for _ in range(10):
                self.sup.step(self.ts)
            print(f"[referee] screenshot -> {shot}", flush=True)
            self.sup.simulationQuit(0)
            return
        while self.sup.step(self.ts) != -1 and not self._done:
            x, y, z, up_z, _ = self._pose()
            t = self.sup.getTime()

            # Падение или переворот
            if z < self.fall["z_min"] or up_z < math.cos(math.radians(self.fall["tilt_max_deg"])):
                self._penalties.append({"type": "fall", "sim_time": round(t, 2), "cost": 0})
                self._final("fall")
                return

            # Ограничения времени
            if t > self.tmo["sim_time_s"]:
                self._final("timeout")
                return
            if time.time() - self._wall_start > self.tmo["wall_clock_s"]:
                self._penalties.append({"type": "wall_clock_hang", "cost": 0})
                self._final("timeout")
                return

            # Здесь проверяем, что объект неподвижен в своей зоне хотя бы 0,4 с
            # Это нужно для открытия двери короткого пути
            # Баллы начисляем в _final по конечному положению объекта
            # Промежуточное попадание в зону не даёт баллов
            if self.man_tasks:
                for task in self.man_tasks:
                    node = self._man_nodes.get(task["object_def"])
                    if node is None:
                        continue
                    px, py, pz = node.getPosition()
                    in_zone = (math.hypot(px - task["x"], py - task["y"]) <= task["radius"]
                               and task.get("z_min", -1e9) <= pz <= task.get("z_max", 1e9))
                    last = task.get("_last"); task["_last"] = (px, py, pz)
                    moved = math.dist((px, py, pz), last) if last else 1.0
                    if in_zone and moved < 0.004:          # Объект стоит на месте, а не проносится через зону
                        task["_settle"] = task.get("_settle", 0.0) + self.ts / 1000.0
                    else:
                        task["_settle"] = 0.0
                    if not task.get("_placed") and task["_settle"] >= 0.4:   # Объект неподвижен в зоне около 0,4 с
                        task["_placed"] = True
                        task["sim_time"] = round(t, 2)
                        placed_n = sum(1 for k in self.man_tasks if k.get("_placed"))
                        print(f"[referee] task {task['name']} placed @ {t:.1f}s "
                              f"[{placed_n}/{len(self.man_tasks)}] (points scored on final rest)", flush=True)
                        if placed_n == len(self.man_tasks) and self._door is not None:
                            self._door_z = 0.0          # Начинаем открывать дверь
                            print("[referee] all placed — door opening", flush=True)
                # Опускаем дверь со скоростью 0,5 м/с, пока она не уйдёт под пол
                if self._door_z is not None and self._door_z < self._door_drop:
                    step = 0.5 * self.ts / 1000.0
                    self._door_z = min(self._door_drop, self._door_z + step)
                    f = self._door.getField("translation")
                    v = f.getSFVec3f()
                    f.setSFVec3f([v[0], v[1], v[2] - step])
                    if self._door_z >= self._door_drop:
                        print("[referee] door OPEN", flush=True)

            # Зачёт контрольных точек
            if self.order == "any":
                # Любая ещё не зачтённая точка засчитывается сразу при входе в её радиус
                for cp in self.cps:
                    if not cp["scored"] and math.hypot(x - cp["x"], y - cp["y"]) <= cp["radius"]:
                        cp["scored"] = True
                        cp["sim_time"] = round(t, 2)
                        self._recolor(cp, GREEN)
                        done = sum(1 for c in self.cps if c["scored"])
                        print(f"[referee] {cp['name']} collected (+{cp['points']}) @ {t:.1f}s "
                              f"[{done}/{len(self.cps)}]", flush=True)
            elif self._idx < len(self.cps):
                cp = self.cps[self._idx]
                if math.hypot(x - cp["x"], y - cp["y"]) <= cp["radius"]:
                    cp["scored"] = True
                    cp["sim_time"] = round(t, 2)
                    self._recolor(cp, GREEN)
                    print(f"[referee] {cp['name']} scored (+{cp['points']}) @ {t:.1f}s", flush=True)
                    self._idx += 1
                    if self._idx < len(self.cps):
                        self._recolor(self.cps[self._idx], YELLOW)   # Подсвечиваем следующую цель

            # Для финиша нужно задержаться внутри кольца, в sequential оно доступно после последней точки
            # В режиме any можно финишировать сразу, но без всех точек баллов будет меньше
            if self.order == "any" or all(c["scored"] for c in self.cps):
                fx, fy, fr = self.finish["x"], self.finish["y"], self.finish["radius"]
                if math.hypot(x - fx, y - fy) <= fr:
                    self._finish_hold += self.ts / 1000.0
                    if self._finish_hold >= self.finish.get("hold_s", 3.0):
                        self._recolor(self.finish, GREEN) if self.finish.get("def") else None
                        self._final("finish")
                        return
                else:
                    self._finish_hold = 0.0


if __name__ == "__main__":
    Referee().run()
