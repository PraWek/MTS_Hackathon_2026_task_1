#!/usr/bin/env python3

import heapq
import math
import os
import sys

import numpy as np

CELL = 0.12
LIDAR_OFFSET = np.array([0.20, 0.0, 0.16])
CAMERA_OFFSET = np.array([0.40, 0.0, 0.07])
PAD_HEIGHT = 0.03
CRUISE = 0.48


def wrap(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


def rotation(roll, pitch, yaw):
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array([[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr],
                     [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr],
                     [-sp, cp*sr, cp*cr]])


class Perception:
    def __init__(self, robot):
        fov, width, self.max_range, layers, vfov = robot.lidar_info()
        self.shape = (layers, width)
        az = fov/2 - (np.arange(width)+.5)*fov/max(width, 1)
        el = np.linspace(vfov/2, -vfov/2, layers) if layers > 1 else np.zeros(layers)
        ce = np.cos(el)[:, None]
        self.rays = np.stack(np.broadcast_arrays(
            ce*np.cos(az), ce*np.sin(az), np.sin(el)[:, None]), axis=-1)
        self.camera = robot.camera_info()

    def scan(self, robot, position, matrix):
        raw = np.asarray(robot.lidar(), dtype=float)
        if raw.size != self.shape[0]*self.shape[1] or not raw.size:
            return None
        raw = raw.reshape(self.shape)
        valid = np.isfinite(raw) & (raw > .30) & (raw < self.max_range)
        if not np.any(valid):
            return None
        body = self.rays*np.where(valid, raw, 0)[..., None] + LIDAR_OFFSET
        world = body @ matrix.T + position
        own = ((body[..., 0] < .34) & (body[..., 0] > -.42)
               & (np.abs(body[..., 1]) < .24) & (body[..., 2] < .10))
        wall = valid & ~own & (world[..., 2] > .12) & (world[..., 2] < 1.2)
        points = world[wall, :2]
        origin = position + matrix @ LIDAR_OFFSET
        horizontal = np.linalg.norm(world[..., :2]-origin[:2], axis=-1)
        nearest = np.min(np.where(wall, horizontal, np.inf), axis=0)
        floor = valid & ~own & (world[..., 2] >= -.06) & (world[..., 2] <= .12)
        floor_range = np.max(np.where(floor, horizontal, 0), axis=0)
        reach = np.minimum(np.where(np.isfinite(nearest), nearest-.12, floor_range), 6.)
        directions = (self.rays[self.shape[0]//2] @ matrix.T)[:, :2]
        directions /= np.maximum(np.linalg.norm(directions, axis=1, keepdims=True), 1e-6)
        samples = np.arange(.12, 6.01, CELL)
        free = origin[:2]+directions[:, None, :]*samples[None, :, None]
        free = free[samples[None, :] < reach[:, None]]
        return points, free

    def markers(self, robot, position, matrix):
        raw = robot.image()
        width, height, fov = self.camera
        if raw is None or not width or not height:
            return []
        pixels = np.frombuffer(raw, dtype=np.uint8)
        if pixels.size != width*height*4:
            return []
        b, g, r = pixels.reshape(height, width, 4)[::2, ::2, :3].astype(float).transpose(2, 0, 1)
        masks = {
            'yellow': (r > 100) & (g > 95) & (r > b*1.25) & (g > b*1.25) & (g > r*.65),
            'green': (g > 85) & (g > r*1.25) & (g > b*1.18),
            'blue': (b > 100) & (b > r*1.3) & (b > g+6) & (g > r*1.1),
            'purple': (b > 90) & (r > 65) & (r > g*1.2) & (b > g*1.3) & (r > b*.55),
        }
        focal = width/(2*math.tan(fov/2))
        origin = position + matrix @ CAMERA_OFFSET
        found = []
        for colour, mask in masks.items():
            for rows, cols in components(mask):
                if len(rows) < 7:
                    continue
                rr, cc = np.asarray(rows), np.asarray(cols)
                light = {'yellow': np.minimum(r, g), 'green': g,
                         'blue': b, 'purple': np.minimum(r, b)}[colour]
                if np.median(light[rr, cc]) < 220:
                    continue
                inside = (rr > 0) & (rr < mask.shape[0]-1) & (cc > 0) & (cc < mask.shape[1]-1)
                rr, cc = rr[inside], cc[inside]
                edge = ~(mask[rr-1, cc] & mask[rr+1, cc] & mask[rr, cc-1] & mask[rr, cc+1])
                rr, cc = rr[edge], cc[edge]
                if len(rr) < 6:
                    continue
                rays = np.column_stack((np.ones(len(rr)), (width/2-.5-2*cc)/focal,
                                        (height/2-.5-2*rr)/focal)) @ matrix.T
                if np.any(rays[:, 2] > -.018):
                    continue
                scale = (PAD_HEIGHT-origin[2])/rays[:, 2]
                xy = origin[:2]+rays[:, :2]*scale[:, None]
                centre, radius, error = fit_circle(xy)
                angles = np.sort(np.arctan2(xy[:, 1]-centre[1], xy[:, 0]-centre[0]))
                coverage = 2*math.pi-np.max(np.diff(np.r_[angles, angles[0]+2*math.pi]))
                distance = np.linalg.norm(centre-position[:2])
                if .13 < radius < .85 and error < .10 and coverage > 1.2 and .3 < distance < 7:
                    found.append((colour, centre, distance))
        return found


def components(mask):
    pending = mask.copy()
    height, width = pending.shape
    for row, col in zip(*np.nonzero(mask)):
        if not pending[row, col]:
            continue
        stack = [(int(row), int(col))]
        pending[row, col] = False
        rows, cols = [], []
        while stack:
            y, x = stack.pop()
            rows.append(y)
            cols.append(x)
            for yy, xx in ((y-1, x), (y+1, x), (y, x-1), (y, x+1)):
                if 0 <= yy < height and 0 <= xx < width and pending[yy, xx]:
                    pending[yy, xx] = False
                    stack.append((yy, xx))
        yield rows, cols


def fit_circle(xy):
    offset = np.mean(xy, axis=0)
    xy = xy-offset
    system = np.column_stack((2*xy, np.ones(len(xy))))
    sol, _, rank, _ = np.linalg.lstsq(system, np.sum(xy*xy, axis=1), rcond=None)
    if rank < 3:
        return offset, 0., math.inf
    radius = math.sqrt(max(0, sol[2]+np.sum(sol[:2]**2)))
    error = float(np.median(np.abs(np.linalg.norm(xy-sol[:2], axis=1)-radius)))
    return sol[:2]+offset, radius, error


class CourseMap:
    def __init__(self):
        self.walls = set()
        self.free = set()
        self.visits = {}
        self.previous_goal = None

    def update(self, points, free, position, dt):
        occupied = set(map(tuple, np.unique(np.floor(points/CELL).astype(int), axis=0)))
        cleared = set(map(tuple, np.unique(np.floor(free/CELL).astype(int), axis=0)))
        # Убираем старые препятствия там, где новый скан видит свободное место
        # Если в ячейку попали и свободный луч, и препятствие, оставляем препятствие
        self.walls.difference_update(cleared)
        self.walls.update(occupied)
        self.free.update(cleared)
        self.free.difference_update(occupied)
        cell = tuple(np.floor(position[:2]/.4).astype(int))
        self.visits[cell] = self.visits.get(cell, 0.)+dt

    def plan(self, position, yaw, target=None):
        size, mid = 101, 50
        anchor = np.floor(position[:2]/CELL).astype(int)-mid
        occupied = np.zeros((size, size), dtype=bool)
        known = np.zeros_like(occupied)
        for source, grid in ((self.walls, occupied), (self.free, known)):
            for wx, wy in source:
                x, y = wx-anchor[0], wy-anchor[1]
                if 0 <= x < size and 0 <= y < size:
                    grid[x, y] = True
        clearance = np.full((size, size), .96)
        for dx in range(-7, 8):
            for dy in range(-7, 8):
                d = math.hypot(dx, dy)*CELL
                if d >= .96:
                    continue
                xs, xe = max(0, dx), min(size, size+dx)
                ys, ye = max(0, dy), min(size, size+dy)
                view = clearance[xs:xe, ys:ye]
                np.minimum(view, np.where(occupied[xs-dx:xe-dx, ys-dy:ye-dy], d, .96), out=view)
        blocked = clearance < .30
        costs = 1+2.2*np.maximum(0, .7-clearance)/.7+np.where(known, 0., 2.5)
        distance = np.full((size, size), np.inf)
        parents = {}
        start = (mid, mid)
        distance[start] = 0
        queue = [(0., mid, mid)]
        neighbours = [(dx, dy, math.hypot(dx, dy)*CELL)
                      for dx in (-1, 0, 1) for dy in (-1, 0, 1) if dx or dy]
        while queue:
            dist, x, y = heapq.heappop(queue)
            if dist > distance[x, y]:
                continue
            for dx, dy, step in neighbours:
                xx, yy = x+dx, y+dy
                if not (1 <= xx < size-1 and 1 <= yy < size-1):
                    continue
                if blocked[xx, yy]:
                    if not (blocked[x, y] and clearance[xx, yy] > clearance[x, y]
                            and math.hypot(xx-mid, yy-mid) < 5):
                        continue
                if dx and dy and (blocked[x+dx, y] or blocked[x, y+dy]):
                    continue
                nd = dist+step*costs[xx, yy]
                if nd < distance[xx, yy]:
                    distance[xx, yy] = nd
                    parents[xx, yy] = (x, y)
                    heapq.heappush(queue, (nd, xx, yy))
        gx, gy = np.indices((size, size))
        world_x, world_y = (gx+anchor[0]+.5)*CELL, (gy+anchor[1]+.5)*CELL
        radial = np.hypot(world_x-position[0], world_y-position[1])
        reachable = np.isfinite(distance) & ~blocked
        if target is not None:
            score = np.hypot(world_x-target[0], world_y-target[1])+.035*distance
            score[~reachable] = np.inf
        else:
            score = np.full((size, size), np.inf)
            for x, y in zip(*np.nonzero(reachable & known & (radial > 1.) & (radial < 4.8))):
                wx, wy = world_x[x, y], world_y[x, y]
                vx, vy = int(math.floor(wx/.4)), int(math.floor(wy/.4))
                visited = sum(self.visits.get((vx+dx, vy+dy), 0)
                              for dx in (-1, 0, 1) for dy in (-1, 0, 1))
                angle = wrap(math.atan2(wy-position[1], wx-position[0])-yaw)
                s = .20*distance[x, y]-.65*radial[x, y]+.65*abs(angle)
                s += 1.8*min(visited, 12)+max(0., .6-clearance[x, y])*3
                if self.previous_goal is not None:
                    s += .25*min(3., math.hypot(wx-self.previous_goal[0], wy-self.previous_goal[1]))
                score[x, y] = s
        if not np.any(np.isfinite(score)):
            return []
        end = tuple(int(i) for i in np.unravel_index(np.argmin(score), score.shape))
        path = [end]
        while path[-1] != start:
            parent = parents.get(path[-1])
            if parent is None:
                return []
            path.append(parent)
        path.reverse()
        result = [(np.array(cell)+anchor+.5)*CELL for cell in path]
        if result:
            if target is not None and np.linalg.norm(result[-1]-target) < .20:
                result.append(np.asarray(target))
            self.previous_goal = result[-1]
        return result


class Navigator:
    def __init__(self, robot):
        self.sensors = Perception(robot)
        self.map = CourseMap()
        self.markers = []
        self.target = None
        self.path = []
        self.last_plan = -math.inf
        self.last_tick = None
        self.command = np.zeros(3)  # vx, vy и скорость поворота
        self.hold_start = None
        self.last_report = -math.inf
        self.scan_start = None
        self.scan_angle = 0.
        self.last_yaw = None
        self.search_pending = True
        self.verify_point = None
        self.verify_started = None
        self.verify_yaw = 0.

    def observe_markers(self, observations, position, now):
        for colour, xy, distance in observations:
            match = min(self.markers, key=lambda m: np.linalg.norm(m['xy']-xy), default=None)
            if match is not None and np.linalg.norm(match['xy']-xy) >= .65:
                match = None
            if match is None:
                match = dict(xy=xy, colour=colour, visited=False, quality=distance, seen=now,
                             yellow_hits=0, last_yellow=-math.inf)
                self.markers.append(match)
            else:
                weight = .35 if distance <= match['quality']+.2 else .06
                match['xy'] = (1-weight)*match['xy']+weight*xy
                match['quality'] = min(match['quality'], distance)
                if not match['visited'] or colour == 'green':
                    match['colour'] = colour
                match['seen'] = now
            if colour == 'green':
                match['visited'] = True
            if colour == 'yellow':
                match['yellow_hits'] = match['yellow_hits']+1 if now-match['last_yellow'] < .35 else 1
                match['last_yellow'] = now
        # Издалека видимая часть площадки может дать две отдельные отметки
        # Когда их координаты сблизятся, объединяем их в одну
        unique = []
        for marker in self.markers:
            duplicate = next((m for m in unique if (m['colour'] == 'purple') == (marker['colour'] == 'purple')
                              and np.linalg.norm(m['xy']-marker['xy']) < .5), None)
            if duplicate is None:
                unique.append(marker)
                continue
            if self.target is marker:
                self.target = duplicate
            duplicate['visited'] |= marker['visited']
            if marker['quality'] < duplicate['quality']:
                duplicate['xy'], duplicate['quality'] = marker['xy'], marker['quality']
            if marker['seen'] > duplicate['seen']:
                duplicate['colour'], duplicate['seen'] = marker['colour'], marker['seen']
            duplicate['last_yellow'] = max(duplicate['last_yellow'], marker['last_yellow'])
            duplicate['yellow_hits'] = max(duplicate['yellow_hits'], marker['yellow_hits'])
        self.markers = unique
        # При последовательном обходе жёлтой бывает только одна площадка
        # Если следующая стала жёлтой, предыдущая засчитана, даже если она уже под камерой
        if self.target is not None and not self.target['visited']:
            next_yellow = any(m is not self.target and m['colour'] == 'yellow'
                              and not m['visited'] and m['yellow_hits'] >= 2
                              and now-m['last_yellow'] < .2 for m in self.markers)
            if (next_yellow and now-self.target['last_yellow'] > .3
                    and np.linalg.norm(self.target['xy']-position[:2]) < .85):
                self.target['visited'] = True
        choices = [m for m in self.markers if m['colour'] == 'yellow' and not m['visited']]
        visited = sum(m['visited'] for m in self.markers if m['colour'] != 'purple')
        if not choices and visited >= 4:
            choices = [m for m in self.markers if m['colour'] == 'purple']
        chosen = self.target if any(m is self.target for m in choices) else min(
            choices, key=lambda m: np.linalg.norm(m['xy']-position[:2]), default=None)
        if chosen is not self.target:
            self.target = chosen
            self.last_plan = -math.inf
            self.hold_start = None
            self.scan_start = None
            self.search_pending = chosen is None
            self.verify_point = None
            self.verify_started = None
        return visited

    def tick(self, robot):
        now = robot.time()
        dt = .1 if self.last_tick is None else min(.25, max(.001, now-self.last_tick))
        self.last_tick = now
        position = np.asarray(robot.position(), dtype=float)
        attitude = np.asarray(robot.imu(), dtype=float)
        if not np.all(np.isfinite(position)) or not np.all(np.isfinite(attitude)):
            self.command[:] = 0
            robot.stop()
            return
        roll, pitch, yaw = attitude
        if now < 1.0 or max(abs(roll), abs(pitch)) > .48:
            self.command[:] = 0
            robot.stop()
            return
        matrix = rotation(roll, pitch, yaw)
        scan = self.sensors.scan(robot, position, matrix)
        if scan is None:
            self.command[:] = 0
            robot.stop()
            return
        points, free = scan
        self.map.update(points, free, position, dt)
        visited = self.observe_markers(self.sensors.markers(robot, position, matrix), position, now)
        goal = self.target['xy'] if self.target is not None else None
        distance = float(np.linalg.norm(goal-position[:2])) if goal is not None else math.inf
        finishing = self.target is not None and self.target['colour'] == 'purple'
        if self.target is not None and not finishing and distance < .18 and self.verify_point is None:
            self.verify_point = goal-1.15*np.array([math.cos(yaw), math.sin(yaw)])
            self.verify_started, self.verify_yaw = now, yaw
            self.last_plan = -math.inf
        planning_goal = self.verify_point if self.verify_point is not None else goal
        if finishing and distance < .12:
            if self.hold_start is None:
                self.hold_start = now
            self.command[:] = 0
            if now-self.hold_start > .6:
                robot.stand()
            else:
                robot.stop()
            return
        self.hold_start = None
        if now-self.last_plan >= .5 and (goal is not None or not self.search_pending):
            self.path = self.map.plan(position, yaw, planning_goal)
            self.last_plan = now
        aim = self.lookahead(position, points)
        if self.target is None and self.search_pending:
            aim = None
        desired = np.zeros(3)
        if self.verify_point is not None:
            # Отходим назад лицом к площадке, чтобы проверить её цвет
            # Одного попадания в расчётный центр недостаточно для зачёта
            point = aim if aim is not None else self.verify_point
            local = (point-position[:2]) @ matrix[:2, :2]
            desired[:2] = np.clip(local*.7, [-.18, -.16], [.18, .16])
            heading = self.verify_yaw if distance < .5 else math.atan2(goal[1]-position[1], goal[0]-position[0])
            desired[2] = np.clip(1.2*wrap(heading-yaw), -.45, .45)
            if np.linalg.norm(self.verify_point-position[:2]) < .16:
                desired[:2] = 0
                if now-self.verify_started > 2 and now-self.target['last_yellow'] < .3:
                    self.verify_point = None
                    self.last_plan = -math.inf
        elif aim is not None:
            delta = aim-position[:2]
            angle = wrap(math.atan2(delta[1], delta[0])-yaw)
            desired[2] = np.clip(1.6*angle, -.75, .75)
            desired[0] = CRUISE*max(0., math.cos(angle))**3
            if abs(angle) > .8:
                desired[0] = min(desired[0], .10)
            if goal is not None and distance < .65:
                local = (goal-position[:2]) @ matrix[:2, :2]
                desired[0] = np.clip(local[0]*.85, -.16, .28)
                desired[1] = np.clip(local[1]*.85, -.16, .16)
                desired[2] = np.clip(angle, -.35, .35) if distance > .35 else 0
        else:
            if self.scan_start is None:
                self.scan_start, self.scan_angle, self.last_yaw = now, 0., yaw
            self.scan_angle += abs(wrap(yaw-self.last_yaw))
            self.last_yaw = yaw
            desired[2] = .45 if self.scan_angle < 2*math.pi else 0.
            if self.scan_angle >= 2*math.pi:
                self.last_plan = -math.inf
                self.scan_start = None
                self.search_pending = False
        limits = np.array([.5, .4, 1.2])*dt
        self.command += np.clip(desired-self.command, -limits, limits)
        self.command = self.choose_safe_command(self.command, points, position, yaw)
        robot.drive(vx=float(self.command[0]), vy=float(self.command[1]), vyaw=float(self.command[2]))
        if now-self.last_report > 8:
            mode = self.target['colour'] if self.target is not None else 'explore'
            print(f'[nav] t={now:.1f} target={mode} confirmed={visited}/4 distance={distance:.2f}', flush=True)
            debug_dir = os.environ.get('GO2_NAV_DEBUG')
            if debug_dir:
                print('pose', position.tolist(), 'imu', attitude.tolist(),
                      'markers', [(m['colour'], m['xy'].tolist(), m['visited']) for m in self.markers], flush=True)
                width, height, _ = robot.camera_info()
                np.savez_compressed(os.path.join(debug_dir, f'frame-{now:07.2f}.npz'),
                                    image=np.frombuffer(robot.image(), dtype=np.uint8).reshape(height, width, 4),
                                    position=position, attitude=attitude, points=points,
                                    path=np.asarray(self.path), cloud=np.asarray(robot.lidar()),
                                    map_walls=np.asarray(list(self.map.walls)),
                                    markers=np.asarray([m['xy'] for m in self.markers]))
            self.last_report = now

    def lookahead(self, position, obstacles):
        if not self.path:
            return None
        distances = [np.linalg.norm(p-position[:2]) for p in self.path]
        nearest = int(np.argmin(distances))
        aim = None
        for point in self.path[nearest+1:]:
            delta = point-position[:2]
            length = np.linalg.norm(delta)
            if length > .85:
                break
            if len(obstacles) and length > .01:
                t = np.clip((obstacles-position[:2]) @ delta/(length*length), 0, 1)
                clearance = np.linalg.norm(obstacles-position[:2]-t[:, None]*delta, axis=1)
                if np.any(clearance < .29):
                    break
            aim = point
        return aim

    @staticmethod
    def choose_safe_command(command, obstacles, position, yaw):
        safe = Navigator.safe_command(command, obstacles, position, yaw)
        if np.allclose(safe, command):
            return safe
        candidates = [safe]
        for alternative in (np.array([command[0], command[1], 0.]),
                            np.array([0., 0., command[2]])):
            candidates.append(Navigator.safe_command(alternative, obstacles, position, yaw))
        weights = np.array([2., 2., .35])
        return min(candidates, key=lambda c: np.sum(weights*(c-command)**2))

    @staticmethod
    def safe_command(command, obstacles, position, yaw):
        if not len(obstacles):
            return command
        cy, sy = math.cos(yaw), math.sin(yaw)
        local = (obstacles-position[:2]) @ np.array([[cy, -sy], [sy, cy]])
        speed = np.linalg.norm(command[:2])
        horizon = .35+speed/.8
        for fraction in np.linspace(.15, 1., 8):
            t = horizon*fraction
            angle = command[2]*t
            ca, sa = math.cos(angle), math.sin(angle)
            if abs(command[2]) > 1e-5:
                w = command[2]
                shift = np.array([(command[0]*sa+command[1]*(ca-1))/w,
                                  (command[0]*(1-ca)+command[1]*sa)/w])
            else:
                shift = command[:2]*t
            future = (local-shift) @ np.array([[ca, -sa], [sa, ca]])
            if np.any((future[:, 0] > -.34) & (future[:, 0] < .43) & (np.abs(future[:, 1]) < .27)):
                return command*max(0., fraction-.15)
        return command


def main():
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   '..', '..', 'common', 'locomotion'))
    from go2_api import Go2
    robot = Go2()
    navigator = Navigator(robot)
    next_tick = 0.
    while robot.step():
        now = robot.time()
        if now >= next_tick:
            navigator.tick(robot)
            next_tick = now+.1


if __name__ == '__main__':
    main()
