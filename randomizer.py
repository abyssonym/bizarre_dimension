from randomtools.tablereader import (
    TableObject, get_global_label, tblpath, addresses, get_random_degree,
    mutate_normal, shuffle_normal)
from randomtools.utils import (
    classproperty, cached_property, get_snes_palette_transformer,
    read_multi, write_multi, utilrandom as random)
from randomtools.interface import (
    get_outfile, get_seed, get_flags, get_activated_codes,
    run_interface, rewrite_snes_meta, clean_and_write, finish_interface)
from collections import defaultdict
from os import path
from time import time
from collections import Counter


VERSION = 0
ALL_OBJECTS = None
DEBUG_MODE = False


def load_areas(area_filename=None):
    if area_filename is None:
        area_filename = path.join(tblpath, "areas.txt")
    area_label = (None, "")
    areadict = defaultdict(set)
    for line in open(area_filename):
        if not line.strip():
            area_label = (None, "")
            continue
        if line.startswith(":"):
            area_label = (None, line[1:])
            continue
        cells = set([])
        for word in line.split():
            if word == "....":
                continue
            cells.add(int(word, 0x10))
        for c in sorted(cells):
            newcells = set([c|0x80, c|1, c|0x81, c|2, c|0x82, c|3, c|0x83])
            assert not cells & newcells
            cells |= newcells

        if isinstance(area_label, tuple):
            area_label = "{0:0>4} {1}".format("%x" % min(cells), area_label[1])
            area_label = area_label.strip()
        assert isinstance(area_label, basestring)
        areadict[area_label] |= set(cells)
    return areadict


class Area:
    def __init__(self, label, cells):
        self.label = label
        self.cells = sorted(cells)
        for ec in self.enemy_cells:
            ec.set_area(self)

    @cached_property
    def enemy_cells(self):
        return [m for m in MapEnemyObject.every if m.index in self.cells]

    @cached_property
    def map_events(self):
        return [me for me in MapEventObject.every
                if me.enemy_cell in self.enemy_cells]

    @cached_property
    def map_sprites(self):
        return [ms for ms in MapSpriteObject.every
                if ms.enemy_cell in self.enemy_cells]

    @classproperty
    def all_areas(self):
        if hasattr(Area, "_all_areas"):
            return Area._all_areas

        areas = load_areas()
        all_areas = []
        for key in sorted(areas):
            cells = areas[key]
            all_areas.append(Area(key, cells))

        Area._all_areas = all_areas

        return Area.all_areas


class Script:
    _all_scripts = []

    def __init__(self, pointer, endpointer=None):
        for s in Script._all_scripts:
            assert s.pointer != pointer
        self.pointer = pointer
        self.endpointer = endpointer
        self.subpointers = set([])
        self.read_script()
        Script._all_scripts.append(self)

    def __eq__(self, other):
        return self.pointer == other.pointer

    def __lt__(self, other):
        assert type(self) is type(other)
        return self.pointer < other.pointer

    @classmethod
    def get_by_pointer(self, pointer):
        if not hasattr(self, "_cached_by_pointer"):
            self._cached_by_pointer = {}
        if pointer in self._cached_by_pointer:
            return self._cached_by_pointer[pointer]

        for s in self._all_scripts:
            if s.pointer == pointer:
                self._cached_by_pointer[pointer] = s
                break
        else:
            s = Script(pointer)
            self._cached_by_pointer[pointer] = s

        return self.get_by_pointer(pointer)

    @classproperty
    def scriptdict(self):
        if hasattr(Script, "_scriptdict"):
            return Script._scriptdict

        scriptdict = {}
        for line in open(path.join(tblpath, "instructions.txt")):
            line = line.strip()
            instruction, description = line.split(':')
            instruction = instruction.strip()
            description = description.strip()
            try:
                key = [int(v, 0x10) for v in instruction.split()[:2]]
            except ValueError:
                key = [int(v, 0x10) for v in instruction.split()[:1]]
            key = tuple(key)

            if key in [(0x09,), (0x1f, 0xc0)]:
                scriptdict[key] = (instruction, description, None)
                continue

            length = len(instruction.split())
            assert key not in scriptdict
            scriptdict[key] = (instruction, description, length)

        Script._scriptdict = scriptdict
        return Script.scriptdict

    def read_script(self):
        f = open(get_outfile(), "r+b")
        pointer = self.pointer
        self.lines = []
        while True:
            f.seek(pointer)
            key = (ord(f.read(1)),)
            if key not in self.scriptdict:
                f.seek(pointer)
                key = tuple(map(ord, f.read(2)))
            if key in self.scriptdict:
                instruction, description, length = self.scriptdict[key]
            else:
                key = (key[0],)
                instruction, description, length = (
                    ("%x" % key[0]).upper(), "ERROR", 1)
                self.scriptdict[key] = (instruction, description, length)

            if "Display Compressed" in description:
                description = "ERROR"
                self.scriptdict[key] = (instruction, description, length)

            if length is None:
                f.seek(pointer+len(key))
                numargs = ord(f.read(1))
                length = len(key) + 1 + (numargs*4)
                # TODO: add subpointers

            f.seek(pointer)
            line = tuple(map(ord, f.read(length)))

            if length is None or key in [
                    (0x06,), (0x08,), (0x0a,),
                    (0x1b, 0x02), (0x1b, 0x03), (0x1f, 0x63)]:
                if length is not None:
                    numargs = 1
                subptrptr = pointer + len(line) - (4*numargs)
                assert subptrptr > pointer
                for i in xrange(numargs):
                    f.seek(subptrptr + (i*4))
                    subpointer = read_multi(f, length=4)
                    self.subpointers.add(subpointer)

            self.lines.append(line)
            pointer += length
            if key == (0x02,) and (
                    self.endpointer is None or pointer >= self.endpointer):
                break
        f.close()
        self.old_length = self.length

    def write_script(self, pointer=None):
        if pointer is None:
            pointer = self.pointer
            assert self.length <= self.old_length
        f = open(get_outfile(), "r+b")
        f.seek(pointer)
        s = ""
        for line in self.lines:
            s += "".join(map(chr, line))
        f.write(s)
        f.close()

    @classmethod
    def get_pretty_line_description(self, line):
        key = (line[0],)
        if key not in self.scriptdict:
            key = tuple(line[:2])
        instruction, description, length = self.scriptdict[key]
        pretty_line = " ".join(["{0:0>2}".format("%x" % v) for v in line])
        return pretty_line, description

    @property
    def pretty_script(self):
        return self.get_pretty_script_full()

    def get_pretty_script(self):
        pretty_lines = []
        descriptions = []
        erroneous = []
        max_length = 0
        for line in self.lines:
            pretty_line, description = self.get_pretty_line_description(line)
            if description == "ERROR":
                erroneous.append(pretty_line.strip())
                continue
            if erroneous:
                pretty_lines.append(" ".join(erroneous))
                descriptions.append("ERROR")
                erroneous = []
            pretty_lines.append(pretty_line)
            descriptions.append(description)
            if "ERROR" not in description:
                max_length = max(max_length, len(pretty_line))
        s = ""
        for pretty_line, description in zip(pretty_lines, descriptions):
            if "ERROR" not in description:
                pretty_line = ("{0:%s}" % (max_length)).format(pretty_line)
                s += "%s : %s\n" % (pretty_line, description)
            else:
                s += pretty_line + "\n"
        return s.strip()

    def get_pretty_script_full(self, exclude_pointers=None):
        if exclude_pointers is None:
            exclude_pointers = []

        s = ("$%x\n" % self.pointer).upper()
        s += self.get_pretty_script()
        subpointers = sorted([p for p in self.subpointers
                              if p not in exclude_pointers])
        exclude_pointers.extend(subpointers)
        for p in subpointers:
            if p & 0xFFC00000 != 0xC00000:
                s += "\n\nINVALID SUB POINTER - %x" % p
            p = p & 0x3FFFFF
            s += "\n\n" + Script.get_by_pointer(p).get_pretty_script_full(
                exclude_pointers=exclude_pointers)
        return s.strip()

    @property
    def length(self):
        return sum([len(line) for line in self.lines])


class GridMixin(object):
    max_rows = 320
    max_columns = 256
    cell_size = 32

    @classproperty
    def height(self):
        assert self.max_rows % self.rows == 0
        return self.max_rows / self.rows

    @classproperty
    def width(self):
        assert self.max_columns % self.columns == 0
        return self.max_columns / self.columns

    @property
    def grid_x(self):
        return self.index % self.columns

    @property
    def grid_y(self):
        return self.index / self.columns

    @property
    def x_bounds(self):
        x1 = self.cell_size * self.grid_x * self.width
        x2 = self.cell_size * (self.grid_x+1) * self.width
        return x1, x2

    @property
    def y_bounds(self):
        y1 = self.cell_size * self.grid_y * self.height
        y2 = self.cell_size * (self.grid_y+1) * self.height
        return y1, y2

    @property
    def center_x(self):
        return sum(self.x_bounds) / 2

    @property
    def center_y(self):
        return sum(self.y_bounds) / 2

    def contains(self, other, lenience=0):
        x11, x12 = self.x_bounds
        y11, y12 = self.y_bounds
        if lenience > 0:
            x11 -= lenience
            x12 += lenience
            y11 -= lenience
            y12 += lenience

        if isinstance(other, tuple):
            x, y = other
            return x11 <= x <= x12 and y11 <= y <= y12

        x21, x22 = other.x_bounds
        y21, y22 = other.y_bounds
        return (x11 <= x21 <= x22 <= x12 and
                y11 <= y21 <= y22 <= y12)

    @classmethod
    def get_by_grid(cls, x, y):
        objs = [o for o in cls.every if o.grid_x == x and o.grid_y == y]
        if len(objs) < 1:
            return None
        assert len(objs) == 1
        return objs[0]

    @classmethod
    def get_by_cell(cls, x, y):
        x = x / cls.width
        y = y / cls.height
        return cls.get_by_grid(x, y)

    @classmethod
    def get_by_pixel(cls, x, y):
        x = x / (cls.width * cls.cell_size)
        y = y / (cls.height * cls.cell_size)
        return cls.get_by_grid(x, y)


class GetByPointerMixin(object):
    @classmethod
    def get_by_pointer(cls, pointer):
        objs = [o for o in cls.every if o.pointer == pointer]
        if len(objs) < 1:
            return None
        assert len(objs) == 1
        return objs[0]


class EventObject(GetByPointerMixin, TableObject):
    def __repr__(self):
        s = "{4:0>5} {0:0>8} {1:0>4} {2:0>4} {3:0>4}".format(*
            ["%x" % v for v in
             [self.event_call, self.event_flag, self.x, self.y, self.pointer]])
        return s

    @property
    def script(self):
        if self.event_call == 0:
            return None
        if self.event_call & 0xFFC00000 != 0xC00000:
            return None
        pointer = self.event_call & 0x3FFFFF
        return Script.get_by_pointer(pointer)

    @property
    def y(self):
        return self.y_facing & 0x3FFF

    @property
    def global_x(self):
        return self.x << 3

    @property
    def global_y(self):
        return self.y << 3


class ZonePositionMixin(object):
    @cached_property
    def zone(self):
        if isinstance(self, MapEventObject):
            candidates = [z for z in ZoneEventObject.every
                          if self.pointer in z.obj_pointers]
        if isinstance(self, MapSpriteObject):
            candidates = [z for z in ZoneSpriteObject.every
                          if self.pointer in z.obj_pointers]
        assert len(candidates) == 1
        return candidates[0]

    @property
    def x_bounds(self):
        return (self.global_x, self.global_x)

    @property
    def y_bounds(self):
        return (self.global_y, self.global_y)

    @cached_property
    def enemy_cell(self):
        me = MapEnemyObject.get_by_pixel(self.global_x, self.global_y)
        assert self.zone.contains(me)
        return me


class MapEventObject(GetByPointerMixin, ZonePositionMixin, TableObject):
    def __repr__(self):
        return "{4:0>4} {0:0>4} {1:0>4} {2:0>2} {3:0>4}".format(
            *["%x" % v for v in [self.global_x, self.global_y, self.event_type,
                                 self.event_index, self.enemy_cell.index]])

    def connect_exit(self, other):
        if other.friend:
            friend = other.friend
        elif other is self:
            friend = self
        friend.event.event_flag = 0

        for x in self.neighbors:
            x.event_index = friend.old_data["event_index"]
            x._connected = True

    @property
    def connected(self):
        if hasattr(self, "_connected"):
            return self._connected
        return False

    @cached_property
    def event(self):
        return EventObject.get_by_pointer(
            0xF0000 | self.old_data["event_index"])

    @property
    def script(self):
        if self.event:
            return self.event.script
        return None

    @cached_property
    def destination_zone(self):
        return ZoneEventObject.get_by_pixel(self.event.global_x,
                                            self.event.global_y)

    @cached_property
    def friend(self):
        candidates = []
        for me in MapEventObject.all_exits:
            mezone = me.destination_zone
            if mezone.muspal_signature == self.zone.muspal_signature:
                candidates.append(me)

        if not candidates:
            return None

        ranker = lambda me: (abs(me.event.global_x - self.global_x) +
                             abs(me.event.global_y - self.global_y))
        ranked_mes = sorted(candidates, key=ranker)
        chosen = ranked_mes[0]
        #distance = ranker(chosen)
        if chosen is self:
            return None
        return chosen

    def get_distance(self, x, y):
        return abs(self.global_x - x) + abs(self.global_y - y)

    @cached_property
    def neighbors(self):
        neighbors = set([
            me for me in MapEventObject.all_exits
            if me.old_data["event_index"] == self.old_data["event_index"]
            and me.zone.muspal_signature == self.zone.muspal_signature
            and self.get_distance(me.global_x, me.global_y) <= 100])
        assert None not in neighbors
        assert self in neighbors
        return neighbors

    @cached_property
    def canonical_neighbor(self):
        neighbors = sorted(self.neighbors, key=lambda n: n.index)
        return neighbors[0]

    @cached_property
    def has_mutual_friend(self):
        return self.friend and self.friend.friend in self.neighbors

    @property
    def is_exit(self):
        return self.event_type == 2

    @classproperty
    def all_exits(self):
        if hasattr(MapEventObject, "_all_exits"):
            return MapEventObject._all_exits

        all_exits = [meo for meo in MapEventObject.every if meo.is_exit]
        MapEventObject._all_exits = all_exits
        return MapEventObject.all_exits

    @classproperty
    def mutual_exits(self):
        if hasattr(MapEventObject, "_mutual_exits"):
            return MapEventObject._mutual_exits

        mutual_exits = [meo for meo in MapEventObject.all_exits
                        if meo.has_mutual_friend]
        MapEventObject._mutual_exits = mutual_exits
        return MapEventObject.mutual_exits

    @property
    def cave_rank(self):
        cluster = Cluster.get_by_exit(self)
        if cluster is not None:
            return cluster.rank
        return None

    @property
    def global_x(self):
        x1, x2 = self.zone.x_bounds
        x = x1 + (self.x*8)
        assert x1 <= x <= x2
        return x

    @property
    def global_y(self):
        y1, y2 = self.zone.y_bounds
        y = y1 + (self.y*8)
        assert y1 <= y <= y2
        return y


class MapSpriteObject(GetByPointerMixin, ZonePositionMixin, TableObject):
    def __repr__(self):
        return "{0:0>2} {1:0>2} {3:0>5} {2:0>4}".format(
            *["%x" % v for v in [self.x, self.y, self.tpt_number,
                                 self.pointer]])

    @property
    def tpt(self):
        return TPTObject.get(self.tpt_number)

    @property
    def script(self):
        pointer = self.tpt.address
        if pointer == 0:
            return None
        assert pointer & 0xFFC00000 == 0xC00000
        return Script.get_by_pointer(pointer & 0x3FFFFF)

    @property
    def global_x(self):
        x1, x2 = self.zone.x_bounds
        x = x1 + self.x
        assert x1 <= x <= x2
        return x

    @property
    def global_y(self):
        y1, y2 = self.zone.y_bounds
        y = y1 + self.y
        assert y1 <= y <= y2
        return y


class TPTObject(TableObject): pass


class MapEnemyObject(GridMixin, TableObject):
    rows = 160
    columns = 128

    def set_area(self, area):
        self._area = area

    @property
    def area(self):
        if hasattr(self, "_area"):
            return self._area
        return None

    @cached_property
    def map_events(self):
        return [me for me in MapEventObject.every if me.enemy_cell is self]

    @cached_property
    def map_sprites(self):
        return [ms for ms in MapSpriteObject.every if ms.enemy_cell is self]

    @property
    def canonical_exit(self):
        if hasattr(self, "_canonical_exit"):
            return self._canonical_exit

        for area in Area.all_areas:
            neighbors = area.enemy_cells
            exits = [meo for meo in MapEventObject.mutual_exits
                     if meo.enemy_cell in neighbors
                     and meo.cave_rank is not None]
            for n in neighbors:
                if not exits:
                    n._canonical_exit = None
                    continue

                measurer = lambda x: ((abs(n.center_x - x.global_x) +
                                       abs(n.center_y - x.global_y)), x.index)

                temp = [x for x in exits if x.enemy_cell is n]
                if not temp:
                    temp = exits
                temp = sorted(temp, key=measurer)

                n._canonical_exit = temp[0]

        assert hasattr(self, "_canonical_exit")
        return self.canonical_exit

    @property
    def cave_rank(self):
        if hasattr(self, "_cave_rank"):
            return self._cave_rank

        chosen = self.canonical_exit
        if chosen is None:
            return None

        rank = chosen.cave_rank
        if rank is not None:
            self._cave_rank = rank
            return self.cave_rank
        return None

    @cached_property
    def palette(self):
        return MapPaletteObject.get_by_grid(
            self.grid_x/4, self.grid_y/2).palette_index

    @cached_property
    def music(self):
        return MapMusicObject.get_by_grid(
            self.grid_x/4, self.grid_y/2).music_index


class MapPaletteObject(GridMixin, TableObject):
    rows = 80
    columns = 32


class MapMusicObject(GridMixin, TableObject):
    rows = 80
    columns = 32


class ZoneMixin(GridMixin):
    rows = 40
    columns = 32

    def __repr__(self):
        s = "%s %x %s" % (self.__class__.__name__, self.index,
                          hex(self.pointer))
        for o in self.objects:
            s += "\n- %s" % o
        return s.strip()

    @property
    def obj_pointers(self):
        if self.pointer == 0:
            return []
        pointers = [(self.pointer + 2 + (self.zone_object.total_size*i))
                    for i in xrange(self.num_objects)]

        '''
        if isinstance(self, ZoneEventObject):
            if self.index == 0x2f0:
                # think there's an error: num_objects should be 2 instead of 3
                pointers = pointers[:2]
            elif self.index == 0x32f:
                pointers = pointers[:4]
            elif self.index == 0x38f:
                pointers = pointers[:1]
        '''

        return pointers

    @cached_property
    def objects(self):
        objects = [self.zone_object.get_by_pointer(p)
                   for p in self.obj_pointers]
        new_objects = []
        for o in objects:
            if o is None or o.x == o.y == 0:
                break
            new_objects.append(o)
        return new_objects


class ZoneEventObject(ZoneMixin, TableObject):
    zone_object = MapEventObject

    @cached_property
    def muspal_signature(self):
        x1, x2 = self.x_bounds
        y1, y2 = self.y_bounds
        me = MapEnemyObject.get_by_pixel(x1, y1)
        return (me.palette, me.music)

    @property
    def exit_report(self):
        print "%x" % self.index, hex(self.pointer)
        for o in self.objects:
            print o
        print


class ZoneSpriteObject(ZoneMixin, TableObject):
    zone_object = MapSpriteObject


class Cluster():
    def __init__(self):
        self.exits = []

    def __eq__(self, other):
        return self.index == other.index

    def __hash__(self):
        return self.index

    def __lt__(self, other):
        if other is None:
            return False
        assert type(self) is type(other)
        return self.rank < other.rank

    @property
    def rank(self):
        if hasattr(self, "_rank"):
            return self._rank
        return None

    @classproperty
    def ranked_clusters(self):
        ranked = sorted([c for c in Cluster._all_clusters
                         if hasattr(c, "_rank")], key=lambda c: c.rank)
        return ranked

    def set_rank(self, rank):
        self._rank = rank

    def set_rank_random(self, rank):
        while True:
            value = random.random()
            if int(rank + value) == rank:
                break
        self.set_rank(rank + value)

    def add_exit(self, s):
        meid, x, y = map(lambda v: int(v, 0x10), s.split())
        candidates = [c for c in MapEventObject.mutual_exits
                      if c.global_x == x and c.global_y == y]
        if len(candidates) < 1:
            raise KeyError
        assert len(candidates) == 1
        chosen = candidates[0].canonical_neighbor
        for x in self.exits:
            if chosen in x.neighbors:
                assert chosen is x
                return
        self.exits.append(chosen)
        self.exits = sorted(self.exits, key=lambda x: x.pointer)

    @classmethod
    def assign_exit_pair(self, a, b):
        if not hasattr(Cluster, "assign_dict"):
            Cluster.assign_dict = {}
        assert a not in self.assign_dict
        assert b not in self.assign_dict
        assert isinstance(a, MapEventObject) and a.is_exit
        assert isinstance(b, MapEventObject) and b.is_exit
        self.assign_dict[a] = b
        self.assign_dict[b] = a

    @property
    def unassigned_exits(self):
        if not hasattr(Cluster, "assign_dict"):
            Cluster.assign_dict = {}
        unassigned = []
        for x in self.exits:
            x = x.canonical_neighbor
            assert x is x.canonical_neighbor
            if x not in Cluster.assign_dict and x not in unassigned:
                unassigned.append(x)
        return unassigned

    @classmethod
    def rank_clusters(self):
        home = Cluster.home
        rank = 0
        home.set_rank_random(rank)
        ranked = set([])
        ranked.add(home)
        while True:
            clusters = sorted(ranked)
            exits = []
            for c in clusters:
                exits.extend(c.exits)

            unranked = []
            for x1 in exits:
                if x1 in Cluster.assign_dict:
                    x2 = Cluster.assign_dict[x1]
                    c = Cluster.get_by_exit(x2)
                    if c not in ranked and c not in unranked:
                        unranked.append(c)
            if not unranked:
                break

            rank += 1
            for u in unranked:
                u.set_rank_random(rank)
                ranked.add(u)
            assert set(unranked) < ranked

    def find_distance(self, other):
        networked = set([self])
        new_networked = set(networked)
        counter = 0
        while True:
            if other in new_networked:
                break
            counter += 1
            temp = set([])
            for n in list(new_networked):
                for x1 in n.exits:
                    x2 = Cluster.assign_dict[x1]
                    c = Cluster.get_by_exit(x2)
                    if c not in networked:
                        networked.add(c)
                        temp.add(c)
            new_networked = temp
        return counter

    @classmethod
    def get_by_exit(self, exit):
        if not hasattr(Cluster, "_exitdict"):
            Cluster._exitdict = {}
        if exit in Cluster._exitdict:
            return Cluster._exitdict[exit]

        clus = [clu for clu in Cluster.generate_clusters()
                if exit in clu.exits]
        assert len(clus) <= 1
        if not clus:
            return None
        Cluster._exitdict[exit] = clus[0]
        return Cluster.get_by_exit(exit)

    @classproperty
    def home(self):
        if hasattr(Cluster, "_home"):
            return Cluster._home

        exits = [me for me in MapEventObject.every if me.is_exit
                 and me.global_y == 0x0150 and me.global_x in (0x1d20, 0x1e78)]
        assert len(exits) == 2
        #exits = [me for me in MapEventObject.every if me.is_exit
        #         and me.global_y == 0x0450 and me.global_x in (0x1f20,)]
        #assert len(exits) == 1
        exits = set(exits)
        chosen = [c for c in Cluster.generate_clusters()
                  if exits <= set(c.exits)]
        assert len(chosen) == 1
        Cluster._home = chosen[0]
        return Cluster.home

    @classproperty
    def goal(self):
        if hasattr(Cluster, "_goal"):
            return Cluster._goal

        exits = [me for me in MapEventObject.every if me.is_exit
                 and me.global_y == 0x07f8 and me.global_x == 0x1088]
        assert len(exits) == 1
        exit = exits[0]
        chosen = [c for c in Cluster.generate_clusters() if exit in c.exits]
        assert len(chosen) == 1
        Cluster._goal = chosen[0]
        return Cluster.goal

    @property
    def index(self):
        if not self.exits:
            return None
        return min(x.pointer for x in self.exits)

    @classmethod
    def generate_clusters(self, filename=None):
        if hasattr(self, "_all_clusters"):
            return self._all_clusters

        if filename is None:
            filename = path.join(tblpath, "exits.txt")

        clu = Cluster()
        all_clusters = []
        for line in open(filename):
            if (line and line[0] in ":#") or not line.strip():
                if clu.exits:
                    all_clusters.append(clu)
                clu = Cluster()
                continue

            try:
                clu.add_exit(line.strip())
            except KeyError:
                pass

        assert len(all_clusters) == len(
            set([clu.index for clu in all_clusters]))
        all_clusters = sorted(all_clusters, key=lambda clu: clu.index)

        self._all_clusters = all_clusters
        return self.generate_clusters()


def generate_cave():
    print "GENERATING CAVE"
    print "Loading clusters..."
    all_clusters = list(Cluster.generate_clusters())

    COMPLETION = 1.0

    def completion_sample(stuff):
        return random.sample(stuff, int(round(len(stuff)*COMPLETION)))

    titanic_ant = [c for c in all_clusters if c.index == 0xf30e2]
    assert len(titanic_ant) == 1
    titanic_ant = titanic_ant[0]
    assert len(titanic_ant.unassigned_exits) == 2

    checkpoints = [Cluster.home, titanic_ant, Cluster.goal]
    for c in checkpoints:
        all_clusters.remove(c)

    print "Categorizing clusters..."
    singletons = [c for c in all_clusters if len(c.unassigned_exits) <= 1]
    pairs = [c for c in all_clusters if len(c.unassigned_exits) == 2]
    multiples = [c for c in all_clusters if len(c.unassigned_exits) >= 3]
    singletons = completion_sample(singletons)
    pairs = completion_sample(pairs)
    multiples = completion_sample(multiples)

    checkpoint_dict = defaultdict(set)
    num_segments = len(checkpoints)-1
    temp_checkpoints = checkpoints[:num_segments]
    random.shuffle(temp_checkpoints)

    candidates = pairs + multiples
    num_per_segment = len(candidates) / num_segments
    num_per_segment = [num_per_segment] * num_segments
    while sum(num_per_segment) < len(candidates):
        max_index = num_segments-1
        index = random.randint(0, max_index)
        num_per_segment[index] += 1
    assert sum(num_per_segment) == len(candidates)
    assert all(nps >= 2 for nps in num_per_segment)

    for i, (num, tc) in enumerate(zip(num_per_segment, temp_checkpoints)):
        for _ in xrange(1000):
            chosens = random.sample(candidates, num)
            chosen_health = sum([len(c.unassigned_exits)-2 for c in chosens])
            if chosen_health >= 3:
                temp = [c for c in candidates if c not in chosens]
                temp_health = sum([len(t.unassigned_exits)-2 for t in chosens])
                threshold = (num_segments-i) * 3
                if temp_health > threshold:
                    break
        else:
            raise Exception("Unable to select appropriate exits.")
        candidates = temp
        checkpoint_dict[tc] |= set(chosens)

    print "Connecting clusters..."
    NONLINEARITY = get_random_degree()
    for cp1, cp2 in zip(checkpoints, checkpoints[1:]):
        chosens = sorted(checkpoint_dict[cp1], key=lambda c: c.index)
        aa = cp1
        assert aa.unassigned_exits
        bb = random.choice(chosens)
        assert bb.unassigned_exits
        a, b = (random.choice(aa.unassigned_exits),
                random.choice(bb.unassigned_exits))
        Cluster.assign_exit_pair(a, b)
        done = [aa, bb]
        chosens = [cp1] + chosens + [cp2]
        while set(done) != set(chosens):
            candidates = [c for c in chosens if c not in done]
            if cp2 in candidates and len(candidates) > 1:
                candidates.remove(cp2)
            bb = random.choice(candidates)
            assert bb.unassigned_exits
            candidates = [d for d in done if d.unassigned_exits]
            max_index = len(candidates)-1
            index = mutate_normal(max_index, 0, max_index, wide=True,
                                  random_degree=NONLINEARITY)
            aa = candidates[index]
            assert aa.unassigned_exits
            a, b = (random.choice(aa.unassigned_exits),
                    random.choice(bb.unassigned_exits))
            Cluster.assign_exit_pair(a, b)
            assert aa in done
            done.append(bb)
        assert cp2 in done

    total_unassigned_exits = []
    for c in checkpoints + pairs + multiples:
        assert not (set(total_unassigned_exits) & set(c.unassigned_exits))
        total_unassigned_exits.extend(c.unassigned_exits)
    assert not (set(singletons) & set(total_unassigned_exits))

    print "Assigning remaining exits..."
    print len(singletons), len(total_unassigned_exits)
    if len(singletons) > len(total_unassigned_exits):
        to_assign = random.sample(singletons, len(total_unassigned_exits))
    else:
        to_assign = list(singletons)
    assert len(to_assign) == len(total_unassigned_exits)

    for s in to_assign:
        assert len(s.unassigned_exits) == 1
        s = s.unassigned_exits[0]
        chosen = random.choice(total_unassigned_exits)
        Cluster.assign_exit_pair(s, chosen)
        total_unassigned_exits.remove(chosen)

    assert not total_unassigned_exits

    Cluster.rank_clusters()

    for cp in checkpoints:
        clusters = sorted(checkpoint_dict[cp])
        unassigned_exits = []
        for c in clusters:
            uxs = c.unassigned_exits
            random.shuffle(uxs)
            unassigned_exits.extend(uxs)
        assert not unassigned_exits

    print int(Cluster.goal.rank), "doors to the finish"
    for clu in Cluster.ranked_clusters:
        for a in clu.exits:
            b = Cluster.assign_dict[a]
            assert b.has_mutual_friend
            a.connect_exit(b)

    for s in singletons:
        if s not in to_assign:
            for x in s.exits:
                assert not x.connected

    for me in MapEventObject.every:
        if me.is_exit and not me.connected:
            #me.connect_exit(me)
            me.connect_exit(Cluster.home.exits[0])

    f = open(get_outfile(), "r+b")
    f.seek(addresses.start_x)
    write_multi(f, 0x1dcc, length=2)
    f.seek(addresses.start_y)
    write_multi(f, 0x0150, length=2)
    f.close()

    s = Script(0x5e70b)
    lines = []
    lines += [
              (0x04, 0x68, 0x00),
              (0x04, 0xC7, 0x00),
              (0x04, 0xC8, 0x00),
              (0x04, 0xA6, 0x01),
              (0x04, 0x05, 0x02),
              (0x05, 0x0B, 0x00),
              (0x1F, 0x11, 0x02),
              (0x1F, 0x11, 0x03),
              (0x1F, 0x11, 0x04),
              (0x1F, 0xB0),
              (0x02,)]
    s.lines = lines
    print s.pretty_script
    s.write_script()


class EnemyObject(TableObject): pass


if __name__ == "__main__":
    try:
        print ("You are using the Earthbound Ancient Cave "
               "randomizer version %s." % VERSION)
        print

        ALL_OBJECTS = [g for g in globals().values()
                       if isinstance(g, type) and issubclass(g, TableObject)
                       and g not in [TableObject]]

        run_interface(ALL_OBJECTS, snes=True)
        hexify = lambda x: "{0:0>2}".format("%x" % x)
        numify = lambda x: "{0: >3}".format(x)
        minmax = lambda x: (min(x), max(x))

        #generate_cave()

        '''
        for me in MapEventObject.every:
            if me.event.event_call == 0:
                continue
            print hex(me.event.event_call)
            script = me.event.script
            print script.pretty_script
            print
            break

        for mso in MapSpriteObject.every:
            if mso.tpt.address == 0:
                continue
            print hex(mso.tpt.address)
            script = mso.script
            print script.pretty_script
            print
            import pdb; pdb.set_trace()
        '''

        for a in Area.all_areas:
            #print "=" * 79
            #print a.label
            for mso in a.map_sprites:
                if mso.tpt.address == 0:
                    continue
                script = mso.script
                script.pretty_script
                '''
                if "Display Shop Menu" in script.pretty_script:
                    print "-" * 79
                    print "SCRIPT %x" % (mso.tpt.address & 0x3FFFFF)
                    print script.get_pretty_script()
                    import pdb; pdb.set_trace()
                '''
            #print "=" * 79
            #print

        for s in sorted(Script._all_scripts):
            pretty_script = s.get_pretty_script()
            if "08 b1 df c5 00" in pretty_script:
                print pretty_script
                print
            elif "08 2f e0 c5 00" in pretty_script:
                print pretty_script
                print
        exit(0)

        print
        print len(Script._all_scripts)
        print len([s for s in sorted(Script._all_scripts) if "Display Shop Menu" in s.pretty_script])
        print len(Script._all_scripts)
        print

        counter = 0
        for s in sorted(Script._all_scripts):
            if "08 b1 df c5 00" in s.get_pretty_script():
                counter += 1
                print s.get_pretty_script()
                print

        print counter
        print

        '''
        for a in Area.all_areas:
            print "=" * 79
            print a.label
            pointers = set([])
            for mso in a.map_events:
                if not (mso.event and mso.event.event_call != 0 and mso.script):
                    continue
                pointers.add(mso.script.pointer)
            for p in sorted(pointers):
                print "-" * 79
                print "SCRIPT %x" % p
                script = Script.get_by_pointer(p)
                print script.pretty_script
            print "=" * 79
            print
        '''

        clean_and_write(ALL_OBJECTS)
        rewrite_snes_meta("EB-AC", VERSION, lorom=False)
        finish_interface()

    except IOError, e:
        print "ERROR: %s" % e
        raw_input("Press Enter to close this program.")
