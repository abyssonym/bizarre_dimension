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
from time import time, sleep
from collections import Counter
from array import array
import json


VERSION = 11
ALL_OBJECTS = None
DEBUG_MODE = False
TEXT_MAPPING = {}
TEXT_INVERSE_MAPPING = []


hexify = lambda x: "{0:0>2}".format("%x" % x)
numify = lambda x: "{0: >3}".format(x)
minmax = lambda x: (min(x), max(x))


text_map_filename = path.join(tblpath, "text_mapping.txt")
for line in open(text_map_filename):
    line = line.strip("\n").strip("\r")
    code, text = line.split("=", 1)
    TEXT_MAPPING[int(code, 0x10)] = text
    TEXT_INVERSE_MAPPING.append((text, [int(a, 0x10) for a in map(''.join, zip(*[iter(code)]*2))]))
TEXT_MAPPING[0] = None


SANCTUARY_BOSS_POINTERS = [
    0x68409,
    0x68410,
    0x6841e,
    0x68417,
    0x68425,
    0x6842c,
    0x68433,
    0x6843a,
    ]

SANCTUARY_HOTSPOT_POINTERS = [
    0x7becf,
    0x7bf95,
    0x7c06e,
    0x7c12a,
    0x7c208,
    0x7c2c6,
    0x7c39b,
    0x7c479,
    ]

SANCTUARY_ACTIVATION_POINTERS = [
    0x9b08d,
    0x9afe1,
    0x9afb6,
    0x9af8b,
    0x9b00c,
    0x9b037,
    0x9b062,
    0x9b0b2,
    ]

SANCTUARY_BOSS_INDEXES = [
    #0x19, 0x1f, 0xc0, 0x147, 0x16e, 0x1d0, 0x30b, 0x3a2]  # NOT 0x2b4
    0x147, 0xc0, 0x16e, 0x1f, 0x1d0, 0x30b, 0x19, 0x3a2]  # ordered, NOT 0x2b4


def bytes_to_text(s):
    result = ""
    while s:
        if len(s) >= 2:
            key = s[:2]
            value = ord(key[1]) | (ord(key[0]) << 8)
        else:
            value = None
        if value not in TEXT_MAPPING:
            key = s[0]
            value = ord(key[0])
        if value not in TEXT_MAPPING:
            raise Exception("Value %x not valid text." % value)
        text = TEXT_MAPPING[value]
        if text is None:
            break
        result += text
        s = s[len(key):]
    return result

def value_to_text(value):
    if value not in TEXT_MAPPING:
        return None
    return TEXT_MAPPING[value]

def values_to_text(values):
    result = ""
    while values:
        if len(values) > 1:
            text = value_to_text(values[0] * 256 + values[1])
            if text is not None:
                result += text
                values = values[2:]
                continue
        text = value_to_text(values[0])
        if text is None:
            raise Exception("Value %x not valid text." % values[0])
        result += text
        values = values[1:]
    return result

def text_to_values(s, allow_multi=True):
    result = []
    mapping = [i for i in TEXT_INVERSE_MAPPING if allow_multi or len(i[0]) == 1]
    while s:
        found = False
        for text, codes in mapping:
            if s.startswith(text):
                result.extend(codes)
                s = s[len(text):]
                found = True
                break
        if found == False:
            raise Exception("String not able to be mapped: %s" % s)
    return tuple(result)
    
def text_to_bytes(s, min_length=25):
    bytelist = list(text_to_values(s, False))
    while len(bytelist) < min_length: 
        bytelist.append(0)
    return array('B', bytelist).tostring()

def ccode_address(address):
    return (address & 0xff, (address & 0xff00) / 0xff,((address + 0xc00000) & 0xff0000) / 0xffff, 0x00)

def ccode_call_address(address):
    return (0x08,) + ccode_address(address)

def ccode_goto_address(address):
    return (0x0a,) + ccode_address(address)

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

        print "Labeling areas..."
        areas = load_areas()
        all_areas = []
        for key in sorted(areas):
            cells = areas[key]
            all_areas.append(Area(key, cells))

        Area._all_areas = all_areas

        return Area.all_areas


class Script:
    _all_scripts = []
    _freespace = (0x1545c0, 0x154fff)

    @classproperty
    def every(cls):
        return Script._all_scripts

    def __init__(self, pointer, endpointer=None):
        self.pointer = pointer
        self.endpointer = endpointer
        self.subpointers = set([])
        if pointer is not None:
            for s in Script._all_scripts:
                assert s.pointer != pointer
            self.read_script()
        Script._all_scripts.append(self)

    def __eq__(self, other):
        return self.pointer == other.pointer

    def __lt__(self, other):
        assert type(self) is type(other)
        return self.pointer < other.pointer

    def __hash__(self):
        return self.pointer

    def replace_sanctuary_boss(self, boss):
        for s in self.subscript_closure:
            if s.is_battle_source:
                newlines = []
                for line in s.lines:
                    if tuple(line[:2]) == (0x1f, 0x23):
                        a, b = (boss.index & 0xFF, boss.index >> 8)
                        newlines.append([0x1f, 0x23, a, b])
                    else:
                        newlines.append(line)
                s.lines = newlines
                s.write_script()

    def replace_item(self, old_item, new_item):
        for s in self.subscript_closure:
            newlines = []
            needs_write = False
            for line in s.lines:
                if (tuple(line[:2]) == (0x1d, 0x00) or tuple(line[:2]) == (0x1d, 0x0e)) and line[3] == old_item.index:
                    needs_write = True
                    newlines.append([line[0], line[1], line[2], new_item.index])
                else:
                    newlines.append(line)
            s.lines = newlines
            if needs_write:
                s.write_script()

    @property
    def is_battle_source(self):
        self.has_battle_trigger
        if hasattr(self, "_is_battle_source"):
            return self._is_battle_source
        return False

    @property
    def has_battle_trigger(self):
        if hasattr(self, "_has_battle_trigger"):
            return self._has_battle_trigger

        self._has_battle_trigger = False

        for line in self.lines:
            if tuple(line[:2]) == (0x1f, 0x23):
                self._is_battle_source = True
                self._has_battle_trigger = True
                break
        else:
            for s in self.subscript_closure:
                if s is self:
                    continue
                if s.has_battle_trigger:
                    self._has_battle_trigger = True
                    break

        return self.has_battle_trigger

    @property
    def is_sanctuary_door(self):
        for line in self.lines:
            if tuple(line[:3]) == (0x1f, 0x66, 0x01):
                index = line[3]
                if index in xrange(0x18, 0x20):
                    return True
        return False

    @property
    def is_swap_safe(self):
        # Currently, only allow scripts with a depth of 1 to be swapped.
        return self._swap_safe and all(len(ss.subscripts) == 0 and ss._swap_safe for ss in self.subscripts)

    def make_sanctuary_door_always_activate(self):
        assert self.is_sanctuary_door
        assert self.lines[0][0] == 0x07
        assert self.lines[1][0] == 0x1b
        self.lines = self.lines[2:]
        self.lines.insert(-1, (0x05, 0x0B, 0x00))   # encounters on
        self.lines.insert(-1, (0x1F, 0x68))         # exit mouse
        assert tuple(self.lines[-1]) == (0x02,)
        self.schedule_for_writing()

    def remove_encounters_off(self):
        if hasattr(self, "_removed_encounters") and self._removed_encounters:
            return
        keys = [(0x04, 0x0b, 0x00),]
        self.remove_instructions(keys, [])
        self._removed_encounters = True

    def remove_exit_mouse_store(self):
        if self.is_sanctuary_door:
            return

        # NOTE: This only targets 0x9B112, apparently
        if hasattr(self, "_removed_exit_mouse") and self._removed_exit_mouse:
            return
        keys = [(0x1f, 0x68),]
        self.remove_instructions(keys, [])
        self._removed_exit_mouse = True

    def remove_teleports(self):
        if hasattr(self, "_removed_teleports") and self._removed_teleports:
            return
        keys = [
            (0x1f, 0x20),
            (0x1f, 0x21),
            (0x1f, 0x69),
            ]
        exceptions = [
            # All post hotel/sleep/bench teleports. Either they normally
            # teleport you to the same map, or they have been manually
            # changed to do so in the TeleportObject cleanup.
            (0x1f, 0x21, 0x02),
            (0x1f, 0x21, 0x0b),
            (0x1f, 0x21, 0x0d),
            (0x1f, 0x21, 0x0e),
            (0x1f, 0x21, 0x11),
            (0x1f, 0x21, 0x13),
            (0x1f, 0x21, 0x27),
            (0x1f, 0x21, 0x2b),
            (0x1f, 0x21, 0x2c),
            (0x1f, 0x21, 0x2e),
            (0x1f, 0x21, 0x2f),
            (0x1f, 0x21, 0x30),
            (0x1f, 0x21, 0x31),
            (0x1f, 0x21, 0x32),
            (0x1f, 0x21, 0x33),
            (0x1f, 0x21, 0x34),
            (0x1f, 0x21, 0x35),
            (0x1f, 0x21, 0x36),
            (0x1f, 0x21, 0x37),
            (0x1f, 0x21, 0x38),
            (0x1f, 0x21, 0x39),
            (0x1f, 0x21, 0x3A),
            (0x1f, 0x21, 0x66),
            (0x1f, 0x21, 0xA2),
            (0x1f, 0x21, 0xC4),
            (0x1f, 0x21, 0xE9), # Unused value - for testing
            ]
        self.remove_instructions(keys, exceptions)
        self._removed_teleports = True

    def remove_party_changes(self):
        if hasattr(self, "_removed_party") and self._removed_party:
            return
        keys = [
            (0x1f, 0x11),
            (0x1f, 0x12),
            ]
        self.remove_instructions(keys, [])
        self._removed_party = True

    def fix_hotels(self):
        if hasattr(self, "_fixed_hotels") and self._fixed_hotels:
            return
        keys = [
            (0x04, 0x7f, 0x01), # Flag that indicates you just slept.
            # Disabling the above flag stops the bug that prevents you from
            # resleeping at a hotel until you luck into flipping it back off,
            # but at some hotels it does cause the music to not return.
            (0x04, 0x02, 0x02), # Flag that indicates you just slept at home.
            ]
        self.remove_instructions(keys, [])
        self._fixed_hotels = True

    def remove_instructions(self, keys, exceptions):
        for s in self.subscript_closure:
            newlines = []
            for line in s.lines:
                if tuple(line[:2]) in keys and tuple(line[:3]) not in exceptions:
                    continue
                if tuple(line[:3]) in keys:
                    continue
                newlines.append(line)
            if len(newlines) < len(s.lines):
                s.lines = newlines
                s.schedule_for_writing()

    def schedule_for_writing(self):
        self.please_write = True

    def fulfill_scheduled_write(self):
        if hasattr(self, "please_write") and self.please_write:
            self.write_script()

    @cached_property
    def subscript_closure(self):
        ss = set([self])
        while True:
            oldlength = len(ss)
            for s in sorted(ss):
                ss |= set(s.subscripts)
            if len(ss) == oldlength:
                break
        return sorted(ss, key=lambda s: s.pointer)

    @cached_property
    def enemy_encounters(self):
        es = []
        for s in self.subscript_closure:
            for line in s.lines:
                if tuple(line[:2]) == (0x1f, 0x23):
                    index = (line[3] << 8) | line[2]
                    e = BattleEntryObject.get(index)
                    es.append(e)
        return sorted(es, key=lambda e: e.index)

    @cached_property
    def items_given(self):
        items = []
        for s in self.subscript_closure:
            for line in self.lines:
                if tuple(line[:2]) == (0x1d, 0x00) or tuple(line[:2]) == (0x1d, 0x0e):
                    i = ItemObject.get(line[3])
                    items.append(i)
        return sorted(items, key=lambda i: i.index)

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

    @property
    def subscripts(self):
        subscripts = []
        for p in sorted(self.subpointers):
            if p & 0xFFC00000 != 0xC00000:
                continue
            p = p & 0x3FFFFF
            s = Script.get_by_pointer(p)
            subscripts.append(s)
        return subscripts

    @property
    def has_shop_call(self):
        return ((0x08, 0xb1, 0xdf, 0xc5, 0x00) in self.lines or
                (0x08, 0x2f, 0xe0, 0xc5, 0x00) in self.lines)

    @property
    def shop_flags(self):
        if not self.properties["shop"]:
            return None

        for s in self.subscripts:
            if s.has_shop_call:
                break
        else:
            for s0 in self.subscripts:
                for s in s0.subscripts:
                    if s.has_shop_call:
                        break
                else:
                    continue
                break
            else:
                return None

        on_flags = set([])
        off_flags = set([])
        for line in s.lines:
            if len(line) == 3 and line[0] == 4:
                on_flags.add((line[2] << 8) | line[1])
            if len(line) == 3 and line[0] == 5:
                off_flags.add((line[2] << 8) | line[1])
        flags = on_flags & off_flags
        if 0x290 in flags:
            flags.remove(0x290)
        return flags

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
            prefix = instruction[:1]
            safe = (prefix == '!')
            instruction = instruction[1:]
            try:
                key = [int(v, 0x10) for v in instruction.split()[:2]]
            except ValueError:
                key = [int(v, 0x10) for v in instruction.split()[:1]]
            key = tuple(key)

            if key in [(0x09,), (0x1f, 0xc0)]:
                scriptdict[key] = (instruction, description, None, safe)
                continue

            length = len(instruction.split())
            assert key not in scriptdict
            scriptdict[key] = (instruction, description, length, safe)

        Script._scriptdict = scriptdict
        return Script.scriptdict

    
    @classproperty
    def newlines(self):
        if hasattr(Script, "_newlines"):
            return Script._newlines

        newlines = []
        newitem = []
        for line in open(path.join(tblpath, "newlines.txt")):
            line = line.strip()
            if len(line) == 0:
                if len(newitem) > 0:
                    newlines.append(newitem)
                    newitem = []
            elif line.startswith("\""):
                line = line.strip("\"")
                newitem.append(text_to_values(line))
            else:
                newitem.append(tuple([int(a, 0x10) for a in line.split()]))
        if len(newitem) > 0:
            newlines.append(newitem)
        Script._newlines = newlines
        return Script.newlines

    def read_script(self):
        f = open(get_outfile(), "r+b")
        pointer = self.pointer
        self._swap_safe = True
        self.lines = []
        nesting = 0
        current_text_line = None
        while True:
            f.seek(pointer)
            key = (ord(f.read(1)),)
            
            if key[0] >= 0x20:
                # Plain text
                if not current_text_line:
                    current_text_line = []
                current_text_line.extend(key)
                pointer += 1
                continue
            
            if key not in self.scriptdict:
                f.seek(pointer)
                key = tuple(map(ord, f.read(2)))
            if key in self.scriptdict:
                instruction, description, length, safe = self.scriptdict[key]
            else:
                key = (key[0],)
                instruction, description, length, safe = (
                    ("%x" % key[0]).upper(), "ERROR", 1, False)
                self.scriptdict[key] = (instruction, description, length, safe)
            
            if "Display Compressed" in description:
                if not current_text_line:
                    current_text_line = []
                f.seek(pointer)
                current_text_line.extend(tuple(map(ord, f.read(length))))
                pointer += length
                continue
                     
            if current_text_line:
                self.lines.append(tuple(current_text_line))
                current_text_line = None

            if safe is False:
                self._swap_safe = False

            if length is None:
                f.seek(pointer+len(key))
                numargs = ord(f.read(1))
                newlength = len(key) + 1 + (numargs*4)
            else:
                newlength = length

            f.seek(pointer)
            line = tuple(map(ord, f.read(newlength)))

            if length is None or key in [
                    (0x06,), (0x08,), (0x0a,),
                    (0x1b, 0x02), (0x1b, 0x03), (0x1f, 0x63), (0x1f, 0x66)]:
                if length is not None:
                    numargs = 1
                subptrptr = pointer + len(line) - (4*numargs)
                assert subptrptr > pointer
                for i in xrange(numargs):
                    f.seek(subptrptr + (i*4))
                    subpointer = read_multi(f, length=4)
                    self.subpointers.add(subpointer)

            self.lines.append(line)
            pointer += newlength
            if key == (0x19, 0x02):
                nesting += 1

            if key == (0x02,) and (
                    self.endpointer is None or pointer >= self.endpointer):
                if nesting == 0:
                    break
                else:
                    nesting -= 1
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
        self.please_write = False

    @classmethod
    def write_new_script(self, lines):
        new_script = Script(None)
        new_script.lines = lines
        if Script._freespace[0] + new_script.length > Script._freespace[1]:
            raise Exception("No free space for new script.")
        new_script.pointer = Script._freespace[0]
        new_script.old_length = new_script.length
        Script._freespace = (Script._freespace[0] + new_script.length, Script._freespace[1])
        new_script.write_script()
        return new_script

    @classmethod
    def get_pretty_line_description(self, line):
        if line[0] >= 0x20 or line[0] in (0x15, 0x16, 0x17):
            # Plain text
            return values_to_text(line), "Plain Text"
        key = (line[0],)
        if key not in self.scriptdict:
            key = tuple(line[:2])
        instruction, description, length, safe = self.scriptdict[key]
        pretty_line = " ".join(["{0:0>2}".format("%x" % v) for v in line])
        return pretty_line, description

    @property
    def pretty_script(self):
        s = self.get_pretty_script_full()
        return s

    @property
    def properties(self):
        if not hasattr(self, "_properties"):
            self.pretty_script
        return self._properties

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
        excluded = (len(subpointers) < len(self.subpointers))
        exclude_pointers.extend(subpointers)
        for p in subpointers:
            if p & 0xFFC00000 != 0xC00000:
                s += "\n\nINVALID SUB POINTER - %x" % p
            p = p & 0x3FFFFF
            s += "\n\n" + Script.get_by_pointer(p).get_pretty_script_full(
                exclude_pointers=exclude_pointers)

        if not (excluded or hasattr(self, "_properties")):
            self._properties = {}
            self._properties["shop"] = "Display Shop Menu" in s
            self._properties["party"] = ("Add Party Member" in s or
                                         "Remove Party Member" in s)
            self._properties["battle"] = "Trigger Battle Scene" in s
            self._properties["flag"] = ("Toggle On Event Flag" in s or
                                        "Toggle Off Event Flag" in s)

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
        if not hasattr(cls, "_by_grid_cache"):
            cls._by_grid_cache = {}

            for o in cls.every:
                assert (o.grid_x, o.grid_y) not in cls._by_grid_cache
                cls._by_grid_cache[o.grid_x, o.grid_y] = o

        if (x, y) in cls._by_grid_cache:
            return cls._by_grid_cache[x, y]

        return None

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


class AncientCave(TableObject):
    flag = 'a'
    flag_description = "with ancient cave mode"

    @classproperty
    def after_order(self):
        return [PsiTeleportObject]

    @classmethod
    def full_randomize(cls):
        AncientCave.class_reseed("ancient")
        generate_cave()
        super(AncientCave, cls).full_randomize()

    @classmethod
    def full_cleanup(cls):
        # Always give ATM Card help text, regardless of flags
        lines = [
            (0x01, ),
            text_to_values("@EarthBound Ancient Cave randomizer version %s." % VERSION),
            (0x03, 0x00),
            text_to_values("@Seed: %s" % get_seed()),
            (0x03, 0x00),
            text_to_values("@Flags: %s" % get_flags()),
            #(0x1f, 0x21, 0xe9), # Teleport to test location 
            (0x13, 0x02)]
        new_atm_help = Script.write_new_script(lines)

        old_atm_help = Script.get_by_pointer(0x5566b)
        old_atm_help.lines = [ccode_goto_address(new_atm_help.pointer)]
        old_atm_help.write_script()

        super(AncientCave, cls).full_cleanup()


class Dialog(TableObject):
    flag = 'd'
    flag_description = "dialogs"

    @classproperty
    def after_order(self):
        return [PsiTeleportObject]

    @classmethod
    def intershuffle(cls):
        cls.class_reseed("inter")
        candidates = [tpt for tpt in TPTObject.every if tpt.script and tpt.script.is_swap_safe]
        shuffled = shuffle_normal(candidates, random_degree=1)

        for a, b in zip(candidates, shuffled):
            a.address = b.old_data["address"]

    @classmethod
    def mutate_all(cls):
        cls.class_reseed("mut")
        pokey_scripts = [
            (Script.get_by_pointer(0x57e1c), 0, 1),
            (Script.get_by_pointer(0x8fb1b), 5, 4),
            (Script.get_by_pointer(0x8fc2e), 4, 4),
            (Script.get_by_pointer(0x8fd11), 5, 4),
            (Script.get_by_pointer(0x8ff31), 4, 4)
        ]
        game_scripts = [tpt.script for tpt in TPTObject.every if tpt.script and tpt.script.is_swap_safe]
        candidates = Script.newlines
        candidates.extend(random.sample(game_scripts, len(candidates) * 3))
        chosen = random.sample(candidates, len(pokey_scripts))

        for (pokey_script, pre_lines, post_lines), new_script in zip(pokey_scripts, chosen):
            if not isinstance(new_script, Script):
                new_script = Script.write_new_script(new_script)
            pointer = new_script.pointer
            call_line = ccode_call_address(pointer)
            new_lines = pokey_script.lines[0:pre_lines]
            new_lines.append(call_line)
            new_lines.extend(pokey_script.lines[-post_lines:])
            pokey_script.lines = new_lines
            pokey_script.write_script()

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

    def door_description(self):
        if not self.is_exit:
            return ("Enemy Cell: %4x Loc: (%4d, %4d) NON-DOOR" %
                (self.enemy_cell.index, self.global_x, self.global_y))
        return ("Enemy Cell: %4x Loc: (%4d, %4d) Dest: (%4d, %4d)" %
            (self.enemy_cell.index, self.global_x, self.global_y, self.new_event.x * 8, self.new_event.y * 8))
    
    def serialize(self):
        result = {
            "index": self.index,
            "enemyCell": self.enemy_cell.index,
            "x": self.global_x,
            "y": self.global_y,
            "onShortestPath": self.on_shortest_path
        }
        if self.is_exit:
            result["xDestination"] = self.new_event.x * 8
            result["yDestination"] = self.new_event.y * 8
        return result

    def connect_exit(self, other, override=False):
        if not override:
            assert not self.connected
            if other.connected:
                assert other.connected is self

        if other is self:
            if self.event.event_flag != 0x8154:
                for x in self.neighbors:
                    x.event_type = 5
        else:
            friend = other.friend
            friend.event.event_flag = 0
            for x in self.neighbors:
                x.event_index = friend.old_data["event_index"]

        for x in self.neighbors:
            x._connected = other

    @classproperty
    def dialogue_doors(cls):
        if hasattr(MapEventObject, "_dialogue_doors"):
            return MapEventObject._dialogue_doors
        MapEventObject._dialogue_doors = [x for x in MapEventObject.all_exits
                                          if x.event.event_flag == 0x8154]
        return MapEventObject.dialogue_doors

    @property
    def connected(self):
        if hasattr(self, "_connected"):
            return self._connected
        return False
        
    @property
    def incoming_exit(self):
        if hasattr(self, "_incoming_exit"):
            return self._incoming_exit
        return False
        
    @property
    def outgoing_exit(self):
        if hasattr(self, "_outgoing_exit"):
            return self._outgoing_exit
        return False
    
    @property
    def on_shortest_path(self):
        if hasattr(self, "_on_shortest_path"):
            return self._on_shortest_path
        return False

    @cached_property
    def event(self):
        return EventObject.get_by_pointer(
            0xF0000 | self.old_data["event_index"])

    @property
    def new_event(self):
        return EventObject.get_by_pointer(
            0xF0000 | self.event_index)

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
        if not hasattr(MapEventObject, "_prelearned_friends"):
            MapEventObject._prelearned_friends = {}
            filename = path.join(tblpath, "meo_friends.txt")
            f = open(filename)
            for line in f:
                if line.startswith("#"):
                    continue
                a, b = line.strip().split()
                a = MapEventObject.get(int(a, 0x10))
                if b == "None":
                    b = None
                else:
                    b = MapEventObject.get(int(b, 0x10))
                self._prelearned_friends[a] = b

        return self._prelearned_friends[self]

        '''
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
        '''

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
        if cluster is not None and cluster.rank is not None:
            max_rank = max(c.rank for c in Cluster.generate_clusters())
            return cluster.rank / max_rank
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
    flag = 'g'
    flag_description = "gift box contents"
    
    @classproperty
    def after_order(self):
        return [PsiTeleportObject, AncientCave]

    def __repr__(self):
        return "{0:0>2} {1:0>2} {3:0>5} {2:0>4}".format(
            *["%x" % v for v in [self.x, self.y, self.tpt_number,
                                 self.pointer]])

    def serialize(self):
        result = {
            "x": self.global_x,
            "y": self.global_y,
        }
        if self.is_money:
            result["money"] = self.money_value
        if self.is_chest and not self.is_money:
            i = self.chest_contents
            result["name"] = i.name
            result["isEquipment"] = i.is_equipment
            result["itemType"] = i.item_type
        if self.is_sanctuary_boss:
            other = [m for m in MapSpriteObject.every if m.old_data["tpt_number"] == self.tpt_number]
            assert len(other) == 1
            result["bossIndex"] = SANCTUARY_BOSS_INDEXES.index(other[0].index)
            result["enemyEncounters"] = self.script.enemy_encounters
        return result    

    @property
    def tpt(self):
        return TPTObject.get(self.tpt_number)

    @property
    def script(self):
        return self.tpt.script

    def set_script(self, other):
        assert isinstance(other, MapSpriteObject)
        self.tpt_number = other.old_data["tpt_number"]

    @property
    def has_battle_trigger(self):
        return self.script and self.script.has_battle_trigger

    @property
    def is_chest(self):
        return self.tpt.is_chest

    @property
    def is_money(self):
        return self.is_chest and ((self.tpt.argument & 0xFF00) >= 0x100)

    @property
    def money_value(self):
        if not self.is_money:
            return None
        return self.tpt.argument - 0x100

    @property
    def chest_contents(self):
        if not self.is_chest:
            return None
        if self.is_money:
            return "MONEY: %s" % self.money_value
        return ItemObject.get(self.tpt.argument)

    @property
    def is_shop(self):
        return self.tpt.is_shop

    @property
    def shop_flag(self):
        if not self.is_shop:
            return None
        shop_flags = self.script.shop_flags
        if shop_flags is None or len(shop_flags) != 1:
            return None
        return sorted(shop_flags)[0]

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

    @property
    def cave_rank(self):
        return self.enemy_cell.cave_rank

    @property
    def is_sanctuary_boss(self):
        return self.index in SANCTUARY_BOSS_INDEXES

    @cached_property
    def nearest_exit(self):
        def dist(x):
            return (((x.global_x - self.global_x)**2) +
                    ((x.global_y - self.global_y)**2))
        candidates = [meo for meo in MapEventObject.all_exits
                      if meo.enemy_cell.area == self.enemy_cell.area]
        sorted_exits = sorted(candidates, key=lambda meo: dist(meo))
        return sorted_exits[0].canonical_neighbor

    @cached_property
    def nearest_cluster(self):
        for c in Cluster.generate_clusters():
            if self.nearest_exit in c.exits:
                return c

    def replace_item(self, old_item, new_item):
        assert(self.chest_contents == old_item)
        self.tpt.argument = new_item.index

    @classmethod
    def mutate_all(cls):
        cls.class_reseed("mut")
        objs = list(cls.every)
        random.shuffle(objs)
        for o in cls.every:
            if hasattr(o, "mutated") and o.mutated:
                continue
            o.reseed(salt="mut")
            o.mutate()
            o.mutate_bits()
            o.mutated = True

    def mutate(self):
        if not self.is_chest:
            return

        if 'a' not in get_flags():
            i = self.chest_contents
            if self.is_money:
                i = ItemObject.get(0x5a) #Hamburger
            if 'wildgifts' in get_activated_codes() or random.random() < 0.2:
                i = random.choice([item for item in ItemObject.every if not item.is_key_item])
            else:
                i = i.get_similar()
            assert self.tpt.argument == self.tpt.old_data["argument"]
            self.tpt.argument = i.index
            return

        # Ancient Cave
        if not hasattr(ItemObject, "done_ones"):
            ItemObject.done_ones = set([])

        cave_rank = self.cave_rank
        if cave_rank is None:
            return

        if random.random() < (cave_rank ** 2):
            candidates = [i for i in ItemObject.ranked
                          if i.rank >= 0 and not i.is_buyable]
        else:
            candidates = [i for i in ItemObject.ranked if i.rank >= 0]

        if (random.random()**4) > cave_rank:
            temp = [c for c in candidates if c.is_equipment]
            if temp:
                cave_rank = cave_rank ** 0.75
                candidates = temp

        candidates = [c for c in candidates
                      if not (c.limit_one and c.index in ItemObject.done_ones)]
        index = int(round(cave_rank * (len(candidates)-1)))
        chosen = candidates[index]
        new_item = chosen.get_similar(candidates=candidates)
        if new_item.name == "Null":
            print "WOAH DOGGY"
            print new_item
            print new_item.rank
            print index
            print candidates

        if new_item.limit_one:
            ItemObject.done_ones.add(new_item.index)
        self.tpt.argument = new_item.index


class SpriteGroupObject(GetByPointerMixin, TableObject):
    def __repr__(self):
        s = "{0:0>4} {1:0>4} {2:0>2} {3:0>2} {4:0>2} ({5:0>2} {6:0>2}) ({7:0>2} {8:0>2}) {9:0>2}".format(*
            ["%x" % v for v in
             [self.index, self.pointer, self.height, self.width, self.size, self.collision_ew_h, self.collision_ew_w, self.collision_ns_h, self.collision_ns_w, self.sprite_count]])
        return s

    
    @property
    def sprite_count(self):
        if self.index == 463:
            return 8
        raw_size = SpriteGroupObject.get(self.index + 1).pointer - self.pointer
        return max(0, (raw_size - 9) / 2)

    def valid_swap(self, other, exclusions):
        if self.index in exclusions or other.index in exclusions:
            return False
        return (self.sprite_count == other.sprite_count
            and self.collision_ns_w == other.collision_ns_w
            and self.collision_ns_h == other.collision_ns_h
            and self.collision_ew_w == other.collision_ew_w
            and self.collision_ew_h == other.collision_ew_h)


class TPTObject(TableObject):
    flag = 'n'
    flag_description = 'npc sprites'

    @property
    def is_chest(self):
        return self.tpt_type == 2

    @property
    def is_shop(self):
        if not self.script:
            return False
        return self.script.properties["shop"]

    @property
    def script(self):
        pointer = self.address
        if pointer == 0:
            return None
        assert pointer & 0xFFC00000 == 0xC00000
        return Script.get_by_pointer(pointer & 0x3FFFFF)

    def mutate(self):
        chests = [33, 195, 214, 233, 262, 322, 408]
        tpt_exclusions = [198] # Meteorite (causes Buzz Buzz scene problems)
        sprite_exclusions = [0, 106, 200, 247, 295, 314, 316, 368,
            369, 371, 373, 374, 375, 376, 381, 410, 420, 428, 430, 431, 439,
            440, 441, 456, 462, 463,
            # Also exclude all chest sprites
            33, 195, 214, 233, 262, 322, 408]

        if self.index in tpt_exclusions:
            return

        if self.sprite in chests:
            self.sprite = random.choice(chests)
            return

        if self.sprite in sprite_exclusions:
            return

        current_sprite = SpriteGroupObject.get(self.sprite)
        candidates = [sg for sg in SpriteGroupObject.every if current_sprite.valid_swap(sg, sprite_exclusions)]
        self.sprite = random.choice(candidates).index
    
    def cleanup(self):
        if 'a' in get_flags() and self.address == 0xc75909:
            self.address = 0xc68017 # Make home phone a normal phone


class PcGfxObject(TableObject):
    flag = 'p'
    flag_description = 'pc sprites'
    
    @classmethod
    def mutate_all(cls):
        cls.class_reseed("mut")
        # Most values from original EarthBound Reshuffler
        # Table order: 0 1 2 3 5 6 4
        potential_pcs = [
            [1, 8, 17, 21, 5, 27, 34, 16, 1, 453],
            [2, 9, 18, 22, 25, 28, 34, 393, 2, 454],
            [3, 10, 19, 23, 25, 29, 34, 394, 3, 3],
            [4, 11, 20, 24, 25, 30, 34, 295, 4, 362],
            [5, 8, 5, 5, 5, 27, 34, 457, 5, 453],
            [6, 8, 6, 6, 5, 27, 34, 16, 6, 453],
            [25, 11, 25, 25, 25, 29, 34, 26, 25, 25],
            [39, 8, 39, 39, 25, 35, 34, 39, 39, 39],
            [44, 10, 44, 44, 48, 29, 34, 394, 48, 44],
            [45, 10, 45, 45, 25, 29, 34, 394, 45, 382],
            [46, 8, 47, 47, 25, 35, 34, 46, 46, 46],
            [51, 8, 51, 51, 51, 35, 34, 51, 51, 51],
            [149, 8, 149, 149, 25, 31, 34, 149, 149, 149],
            [150, 8, 150, 150, 25, 32, 34, 150, 150, 150],
            [182, 10, 182, 182, 25, 29, 34, 357, 182, 182],
            [435, 8, 435, 435, 25, 27, 34, 435, 435, 435]]
        if 'funsize' in get_activated_codes():
            potential_pcs = [
                [1, 8, 17, 21, 5, 27, 34, 16, 1, 453],
                [27, 34, 27, 27, 27, 1, 8, 16, 1, 453],
                [2, 9, 18, 22, 25, 28, 34, 393, 2, 454],
                [3, 10, 19, 23, 25, 29, 34, 394, 3, 3],
                [4, 11, 20, 24, 25, 30, 34, 295, 4, 362],
                [5, 457, 5, 5, 5, 27, 34, 457, 5, 453],
                [6, 8, 6, 6, 5, 27, 34, 16, 6, 453],
                [25, 26, 25, 25, 25, 29, 34, 26, 25, 25],
                [39, 8, 39, 39, 25, 35, 34, 39, 39, 39],
                [40, 8, 42, 43, 25, 399, 34, 359, 359],
                [44, 10, 44, 44, 458, 29, 34, 394, 48, 44],
                [45, 10, 45, 45, 25, 29, 34, 394, 45, 382],
                [46, 8, 47, 47, 25, 35, 34, 46, 46, 46],
                [51, 8, 51, 51, 51, 35, 34, 51, 51, 51],
                [149, 8, 149, 149, 25, 31, 34, 149, 149, 149],
                [150, 8, 150, 150, 25, 32, 34, 150, 150, 150],
                [182, 10, 182, 182, 25, 29, 34, 357, 182, 182],
                [435, 8, 435, 435, 25, 27, 34, 435, 435, 435]]
        new_pcs = random.sample(potential_pcs, 4)

        for index in [0, 1, 2, 3]: # Only randomize 4 main PCs
            o = cls.get(index)
            new_pc = new_pcs[index]
            o.default   = new_pc[0]
            o.dead      = new_pc[1]
            o.ladder    = new_pc[2]
            o.rope      = new_pc[3]
            o.tiny      = new_pc[5]
            o.tiny_dead = new_pc[6]
            o.robot     = new_pc[4]


class MapEnemyObject(GridMixin, TableObject):
    flag = 'a'

    rows = 160
    columns = 128

    def serialize(self):
        result = {
            "index": self.index,
            "xBounds": self.x_bounds,
            "yBounds": self.y_bounds,
            "caveRank": self.cave_rank,
            "canonicalExit": self.canonical_exit.index if self.canonical_exit else None,
            "enemyGroup": self.enemy_group,
            "area": self.area.label
        }
        return result

    def set_area(self, area):
        self._area = area

    @classproperty
    def after_order(self):
        return [AncientCave]

    @property
    def area(self):
        if not hasattr(self, "_area"):
            Area.all_areas

        return self._area

    @property
    def neighbors(self):
        neighbors = []
        for y in xrange(-1, 2):
            for x in xrange(-1, 2):
                index = self.index + x + (y * self.columns)
                try:
                    neighbors.append(MapEnemyObject.get(index))
                except KeyError:
                    continue

        for n in list(neighbors):
            if abs(n.grid_x - self.grid_x) > 1:
                neighbors.remove(n)
                continue
            if n.area and self.area and n.area.label != self.area.label:
                neighbors.remove(n)

        return neighbors

    @cached_property
    def enemy_adjacent(self):
        if self.old_data["enemy_place_index"] > 0:
            return True
        return any([n.old_data["enemy_place_index"] > 0
                    for n in self.neighbors])

    @cached_property
    def map_events(self):
        return [me for me in MapEventObject.every if me.enemy_cell is self]

    @cached_property
    def map_sprites(self):
        return [ms for ms in MapSpriteObject.every if ms.enemy_cell is self]

    @property
    def canonical_exit(self):
        if not hasattr(MapEnemyObject, "_prelearned_canonical_exits"):
            MapEnemyObject._prelearned_canonical_exits = {}
            filename = path.join(tblpath, "meo_canonical_exits.txt")
            f = open(filename)
            for line in f:
                a, b = line.strip().split()
                a = MapEnemyObject.get(int(a, 0x10))
                if b == "None":
                    b = None
                else:
                    b = MapEventObject.get(int(b, 0x10))
                self._prelearned_canonical_exits[a] = b

        return self._prelearned_canonical_exits[self]

        '''
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
        '''

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

    @property
    def enemy_group(self):
        return EnemyPlaceObject.get(self.enemy_place_index)

    @cached_property
    def palette(self):
        return MapPaletteObject.get_by_grid(
            self.grid_x/4, self.grid_y/2).palette_index

    @cached_property
    def music(self):
        return MapMusicObject.get_by_grid(
            self.grid_x/4, self.grid_y/2).music_index

    def cave_sanitize_events(self):
        for o in self.map_events + self.map_sprites:
            script = o.script
            if script is None:
                continue
            script.remove_exit_mouse_store()
            script.remove_encounters_off()
            script.remove_teleports()
            script.remove_party_changes()
            script.fix_hotels()

    def randomize(self):
        assert 'a' in get_flags()

        # ANCIENT CAVE
        if self.cave_rank is None:
            return

        if not self.enemy_adjacent and random.random() > get_random_degree():
            return

        if random.random() > 0.1:
            self.enemy_place_index = 0
            return

        if self.old_data["enemy_place_index"] == 0 and self.map_sprites:
            return

        if random.random() < 0.01:
            # magic butterfly
            self.enemy_place_index = (
                EnemyPlaceObject.valid_ranked_placements[0].index)
            return

        max_index = len(EnemyPlaceObject.valid_ranked_placements)-1
        index = int(round(max_index * (self.cave_rank**1.5)))
        index = max(index, 1)

        chosen = EnemyPlaceObject.valid_ranked_placements[index]
        chosen = chosen.get_similar()
        self.enemy_place_index = chosen.index

    def cleanup(self):
        if "easymodo" in get_activated_codes():
            self.enemy_place_index = 0


class MapPaletteObject(GridMixin, TableObject):
    rows = 80
    columns = 32


class MapMusicObject(GridMixin, TableObject):
    rows = 80
    columns = 32

class TeleportObject(TableObject):
    flag = 'a'

    def cleanup(self):
        if 'a' in get_flags():
            if self.index == 0x02: # Fourside hotel
                self.x = 748
                self.y = 764
            if self.index == 0x0B: # Onett hotel
                self.x = 1004
                self.y = 188
            if self.index == 0x11: # Threed hotel
                self.x = 842
                self.y = 1164
            if self.index == 0x13: # Twoson hotel
                self.x = 948
                self.y = 908
            if self.index == 0x27: # Happy Happy hotel
                self.x = 879
                self.y = 206
            if self.index == 0x32: # Tenda hotel
                self.x = 55
                self.y = 18
            if self.index == 0x0E: # Summers hotel
                self.x = 838
                self.y = 1182
            if self.index == 0xA2: # Moonside hotel
                self.x = 817
                self.y = 733
            if self.index == 0xC4: # Ness's house
                self.x = 954
                self.y = 45
            if self.index == 0xE9: # Unused value - for testing
                self.x = 57
                self.y = 292

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
        self.optional = False

    def __repr__(self):
        s = "%s %6x - Rank %3d" % (self.__class__.__name__, self.index, self.rank or 0)
        for o in self.exits:
            s += "\n- %s" % o.door_description()
        return s.strip()

    def __eq__(self, other):
        return self.index == other.index

    def __hash__(self):
        return self.index

    def __lt__(self, other):
        if other is None:
            return False
        assert type(self) is type(other)
        return self.rank < other.rank

    def serialize(self):
        result = {
            "index": self.index,
            "rank": self.rank,
            "onShortestPath": self.on_shortest_path,
            "explicitBounds": {
                "x1": min([ec.x_bounds[0] for ec in self.explicit_enemy_cells]),
                "x2": max([ec.x_bounds[1] for ec in self.explicit_enemy_cells]),
                "y1": min([ec.y_bounds[0] for ec in self.explicit_enemy_cells]),
                "y2": max([ec.y_bounds[1] for ec in self.explicit_enemy_cells])
            },
            "areaBounds": {
                "x1": min([ec.x_bounds[0] for ec in self.enemy_cells]),
                "x2": max([ec.x_bounds[1] for ec in self.enemy_cells]),
                "y1": min([ec.y_bounds[0] for ec in self.enemy_cells]),
                "y2": max([ec.y_bounds[1] for ec in self.enemy_cells])
            },
            "doors": self.exits
        }
        return result

    def generate_image(self, filename=None, chosen_exit=None):
        from subprocess import call
        x1 = min([ec.x_bounds[0] for ec in self.enemy_cells])
        x2 = max([ec.x_bounds[1] for ec in self.enemy_cells])
        y1 = min([ec.y_bounds[0] for ec in self.enemy_cells])
        y2 = max([ec.y_bounds[1] for ec in self.enemy_cells])
        cropstring = "%sx%s+%s+%s!" % (x2-x1, y2-y1, x1, y1)
        cmd = ["convert", "fullmap.png", "-crop", cropstring]
        if chosen_exit is not None:
            assert chosen_exit in self.exits
            assert x1 <= chosen_exit.global_x <= x2
            assert y1 <= chosen_exit.global_y <= y2
            x3 = chosen_exit.global_x - x1
            y3 = chosen_exit.global_y - y1
            dimensions = "%s,%s %s,%s" % (x3, y3, x3, y3+5)
            cmd += ["-fill", "red", "-stroke", "black",
                    "-draw", "circle %s" % dimensions]

        cmd.append(filename)
        print " ".join(cmd)
        call(cmd)

    @classmethod
    def find_shortest_path(cls, home=None, goal=None):
        if home is None:
            home = Cluster.home
        if goal is None:
            goal = Cluster.goal

        leafs = [home]
        shortest_paths = {home: []}
        while True:
            new_leafs = []
            for leaf in leafs:
                options = [leaf.get_connected_cluster(x) for x in leaf.exits]
                for o in options:
                    if o in shortest_paths:
                        continue
                    shortest_paths[o] = shortest_paths[leaf] + [leaf]
                    new_leafs.append(o)
            leafs = new_leafs
            if goal in shortest_paths:
                return shortest_paths[goal] + [goal]

    @classmethod
    def mark_shortest_path(cls):
        shortest_path = Cluster.find_shortest_path()
        for (i, c) in enumerate(shortest_path):
            c._on_shortest_path = True
            try:
                c2 = shortest_path[i+1]
                for x in c.exits:
                    if c.get_connected_cluster(x) is c2:
                        x._on_shortest_path = True
                        break
                else:
                    raise Exception("Break in path??")
            except IndexError:
                break

    @classmethod
    def generate_map(cls):
        shortest_path = Cluster.find_shortest_path()
        for (i, c) in enumerate(shortest_path):
            filename = "maps/{0}_{1:0>3}.png".format(get_seed(), i)
            try:
                c2 = shortest_path[i+1]
                for x in c.exits:
                    if c.get_connected_cluster(x) is c2:
                        break
                else:
                    raise Exception("Break in path??")
            except IndexError:
                x = None
            c.generate_image(filename, chosen_exit=x)


    def get_connected_cluster(self, x1):
        x2 = Cluster.assign_dict[x1]
        c = Cluster.get_by_exit(x2)
        return c

    @property
    def area(self):
        return self.exits[0].enemy_cell.area

    @property
    def enemy_cells(self):
        return self.area.enemy_cells

    @property
    def explicit_enemy_cells(self):
        return [exit.enemy_cell for exit in self.exits]

    @property
    def map_sprites(self):
        return self.area.map_sprites

    @property
    def has_battle_trigger(self):
        return any([ms.has_battle_trigger for ms in self.map_sprites])

    @property
    def rank(self):
        if hasattr(self, "_rank"):
            return self._rank
        return None
    
    @property
    def on_shortest_path(self):
        if hasattr(self, "_on_shortest_path"):
            return self._on_shortest_path
        return False

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
        if s.startswith("."):
            self.optional = True
            s = s[1:]

        force = False
        if s.startswith("!"):
            force = True
            s = s[1:]
            
        incoming = False
        if s.startswith("("):
            incoming = True
            s = s[1:]

        outgoing = False
        if s.startswith(")"):
            outgoing = True
            s = s[1:]

        meid, x, y = map(lambda v: int(v, 0x10), s.split())
        if not force:
            candidates = [c for c in MapEnemyObject.get(meid).map_events
                          if c.is_exit and c.has_mutual_friend and
                          c.global_x == x and c.global_y == y]
        else:
            candidates = [c for c in MapEnemyObject.get(meid).map_events
                          if c.is_exit and c.global_x == x and c.global_y == y]
        if len(candidates) < 1:
            raise KeyError
        assert len(candidates) == 1
        chosen = candidates[0].canonical_neighbor
        for x in self.exits:
            if chosen in x.neighbors:
                assert chosen is x
                return

        chosen.force = False
        if force:
            chosen.force = True
        chosen._incoming_exit = incoming
        chosen._outgoing_exit = outgoing

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

    @property
    def unassigned_incoming_exits(self):
        unassigned = self.unassigned_exits
        incoming = [e for e in unassigned if e.incoming_exit]
        if len(incoming) > 0:
            return incoming
        return unassigned

    @property
    def unassigned_outgoing_exits(self):
        unassigned = self.unassigned_exits
        outgoing = [e for e in unassigned if e.outgoing_exit]
        if len(outgoing) > 0:
            return outgoing
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

        #exits = [me for me in MapEventObject.every if me.is_exit
        #         and me.global_y == 0x0150 and me.global_x in (0x1d20, 0x1e78)]
        #assert len(exits) == 2
        exits = [me for me in MapEventObject.every if me.is_exit
                 and me.global_y == 0x0450 and me.global_x in (0x1f20,)]
        assert len(exits) == 1
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

        print "Loading clusters..."
        clu = Cluster()
        all_clusters = []
        for line in open(filename):
            if line and line[0] == "#":
                continue

            if (line and line[0] == ":") or not line.strip():
                if clu.exits:
                    all_clusters.append(clu)
                clu = Cluster()
                continue

            try:
                clu.add_exit(line.strip())
            except KeyError:
                pass
                #print line.strip()
                #import pdb; pdb.set_trace()

        assert len(all_clusters) == len(
            set([clu.index for clu in all_clusters]))
        all_clusters = sorted(all_clusters, key=lambda clu: clu.index)

        self._all_clusters = all_clusters
        return self.generate_clusters()


def generate_cave():
    print "GENERATING CAVE"
    all_clusters = list(Cluster.generate_clusters())

    COMPLETION = 1.0

    def completion_sample(stuff):
        return random.sample(stuff, int(round(len(stuff)*COMPLETION)))

    titanic_ant = [c for c in all_clusters if c.index == 0xf30e2]
    assert len(titanic_ant) == 1
    titanic_ant = titanic_ant[0]
    assert len(titanic_ant.unassigned_exits) == 2

    sbosses = [mso for mso in MapSpriteObject.every
               if mso.index in SANCTUARY_BOSS_INDEXES]
    sclusters = [mso.nearest_cluster for mso in sbosses]
    random.shuffle(sclusters)

    # Remove the middle door from the Electro Specter cluster.
    # This turns this cluster into a two-exit cluster, where the left exit 
    # arrives from the middle hole and departs from the left hole.
    # This prevents non-Euclidian layouts and can guarantee the boss cannot be 
    # skipped if incoming-outgoing preferences are set on the remaining doors.
    # This must be done after the above section to find the boss cluster.
    electro_specter = [c for c in all_clusters if c.index == 0xf2696]
    assert len(electro_specter) == 1
    electro_specter = electro_specter[0]
    bad_exit = [e for e in electro_specter.exits if e.enemy_cell.index == 0x0138]
    assert len(bad_exit) == 1
    electro_specter.exits.remove(bad_exit[0])

    checkpoints = [Cluster.home] + sclusters + [Cluster.goal]
    for c in checkpoints:
        all_clusters.remove(c)

    print "Categorizing clusters..."
    singletons = [c for c in all_clusters if len(c.unassigned_exits) <= 1]
    pairs = [c for c in all_clusters if len(c.unassigned_exits) == 2]
    multiples = [c for c in all_clusters if len(c.unassigned_exits) >= 3]

    if COMPLETION < 1.0:
        singletons = completion_sample(singletons)
        pairs = completion_sample(pairs)
        multiples = completion_sample(multiples)

    AncientCave.class_reseed("selection")

    checkpoint_dict = defaultdict(set)
    num_segments = len(checkpoints)-1
    temp_checkpoints = checkpoints[1:num_segments]
    random.shuffle(temp_checkpoints)
    temp_checkpoints.insert(0, Cluster.home)

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
            if i == 0:
                temp_candidates = [
                    c for c in candidates if not c.has_battle_trigger]
                chosens = random.sample(temp_candidates, num)
            else:
                chosens = random.sample(candidates, num)
            chosen_health = sum([len(c.unassigned_exits)-2 for c in chosens])
            if chosen_health >= 3:
                temp = [c for c in candidates if c not in chosens]
                temp_health = sum([len(t.unassigned_exits)-2 for t in temp])
                threshold = ((num_segments-1)-i) * 3
                if temp_health >= threshold:
                    candidates = temp
                    break
        else:
            raise Exception("Unable to select appropriate exits.")
        assert tc not in checkpoint_dict
        checkpoint_dict[tc] |= set(chosens)

    AncientCave.class_reseed("connections")

    print "Connecting clusters..."
    NONLINEARITY = get_random_degree()
    for cp1, cp2 in zip(checkpoints, checkpoints[1:]):
        chosens = sorted(checkpoint_dict[cp1], key=lambda c: c.index)
        aa = cp1
        assert aa.unassigned_exits

        assert any([len(c.exits) > 2 for c in chosens])
        if cp1 is Cluster.home or random.choice([True, False]):
            candidates = [c for c in chosens if len(c.unassigned_exits) > 2]
        else:
            candidates = [c for c in chosens if len(c.unassigned_exits) > 1]
        candidates = sorted(candidates,
            key=lambda c: (len(c.exits), random.random(), c.index))
        max_index = len(candidates)-1
        index = random.randint(random.randint(0, max_index), max_index)
        bb = candidates[index]
        assert bb in candidates
        assert bb in chosens
        assert bb.unassigned_exits
        a, b = (random.choice(aa.unassigned_outgoing_exits),
                random.choice(bb.unassigned_incoming_exits))
        Cluster.assign_exit_pair(a, b)
        assert bb.unassigned_exits
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
            if len(candidates) == 1 and len(aa.unassigned_exits) == 1 and len(bb.unassigned_exits) == 1:
                raise Exception("Something weird here.")
            a, b = (random.choice(aa.unassigned_outgoing_exits),
                    random.choice(bb.unassigned_incoming_exits))
            Cluster.assign_exit_pair(a, b)
            assert aa in done
            done.append(bb)
        assert cp2 in done

    AncientCave.class_reseed("filling")

    total_unassigned_exits = []
    for c in checkpoints + pairs + multiples:
        assert not (set(total_unassigned_exits) & set(c.unassigned_exits))
        total_unassigned_exits.extend(c.unassigned_exits)
    assert not (set(singletons) & set(total_unassigned_exits))

    print "Assigning remaining exits..."
    to_assign = [s for s in singletons if not s.optional]
    assert len(to_assign) < len(total_unassigned_exits)
    remaining_singletons = [s for s in singletons if s.optional]
    while len(to_assign) < len(total_unassigned_exits):
        x = random.choice(remaining_singletons)
        remaining_singletons.remove(x)
        to_assign.append(x)
    assert len(to_assign) == len(total_unassigned_exits)

    for s in to_assign:
        assert len(s.unassigned_exits) == 1
        s = s.unassigned_exits[0]
        chosen = random.choice(total_unassigned_exits)
        Cluster.assign_exit_pair(s, chosen)
        total_unassigned_exits.remove(chosen)

    assert not total_unassigned_exits
    Cluster.rank_clusters()

    # do AFTER ranking clusters
    AncientCave.class_reseed("bosses")
    replace_sanctuary_bosses()

    print int(Cluster.goal.rank), "doors to the finish"
    for clu in Cluster.ranked_clusters:
        for a in clu.exits:
            b = Cluster.assign_dict[a]
            assert b.force or b.has_mutual_friend
            a.connect_exit(b)

    for s in singletons:
        if s not in to_assign:
            for x in s.exits:
                assert not x.connected

    for me in MapEventObject.every:
        if me.is_exit and not me.connected:
            me.connect_exit(me)
            #me.connect_exit(Cluster.home.exits[0])

    if "giygastest" in get_activated_codes():
        homex = Cluster.home.exits[0]
        goalx = Cluster.goal.exits[0]
        homex.connect_exit(goalx, override=True)
        goalx.connect_exit(homex, override=True)

    f = open(get_outfile(), "r+b")
    f.seek(addresses.start_x)
    write_multi(f, 0x1f80, length=2)
    f.seek(addresses.start_y)
    write_multi(f, 0x0450, length=2)
    f.close()

    s = Script(0x5e70b)
    lines = []
    lines += [
        (0x04, 0x58, 0x00),     # enable Winters phones
        (0x04, 0x62, 0x00),     # enable home phone
        (0x04, 0x68, 0x00),     # normal music in onett
        (0x04, 0xC7, 0x00),     # know dad's phone number
        (0x04, 0xC8, 0x00),     # know mom's phone number
        (0x04, 0xC9, 0x00),     # know escargo express phone number
        (0x04, 0xA6, 0x01),     # daytime in onett
        (0x04, 0x05, 0x02),     # turn on lights at home
        (0x04, 0xD5, 0x01),     # Mom heal part 1
        (0x04, 0x5E, 0x00),     # Mom heal part 2

        #(0x04, 0x74, 0x01),     # become robots

        (0x05, 0x0B, 0x00),     # "enemies won't appear" flag (off)
        (0x1F, 0x11, 0x02),     # recruit paula
        (0x1F, 0x11, 0x03),     # recruit jeff
        (0x1F, 0x11, 0x04),     # recruit poo
        (0x1F, 0x68),           # store exit mouse coordinates
        (0x1F, 0xB0),           # save the game
        (0x02,)]
    s.lines = lines
    s.write_script()

    for sap in SANCTUARY_ACTIVATION_POINTERS:
        script = Script.get_by_pointer(sap)
        script.make_sanctuary_door_always_activate()

    exit_mouse = Script(0x2f9ef4)
    for i, line in enumerate(exit_mouse.lines):
        if tuple(line) == (0x1f, 0x41, 0x05):
            exit_mouse.lines = exit_mouse.lines[i:]
    # TODO: Make exit mouse reuseable?
    exit_mouse.lines.insert(0, (0x05, 0x0B, 0x00))          # encounters on
    exit_mouse.lines.insert(0, (0x1d, 0x01, 0xff, 0xc5))    # one use only
    exit_mouse.write_script()

    print "Sanitizing cave events..."
    # Special Events
    # Giygas - flags necessary to be set upon entering for battle to function
    giygas_enter = Script.get_by_pointer(0x9af3a)
    assert tuple(giygas_enter.lines[0]) == (
        0x06, 0x49, 0x00, 0x2f, 0x99, 0xc9, 0x00)
    giygas_enter.lines[0] = (0x04, 0x74, 0x01)
    giygas_enter.write_script()
    
    # Mom in Ness's house - manual changes to get to heal state
    mom_talk = Script.get_by_pointer(0x750e3)
    assert tuple(mom_talk.lines[0]) == (
        0x06, 0x49, 0x00, 0x7d, 0x54, 0xc7, 0x00)
    mom_talk.lines[0] = (0x0a, 0x24, 0x51, 0xc7, 0x00)
    mom_talk.write_script()

    # Chaos Theater - remove show, due to panning bug when PC sprites randomized
    if 'p' in get_flags():
        chaos_show_trigger = Script.get_by_pointer(0x99fe0)
        assert tuple(chaos_show_trigger.lines[0]) == (
            0x06, 0x3f, 0x01, 0x2f, 0x99, 0xc9, 0x00)
        chaos_show_trigger.lines[0] = (0x0a, 0x2f, 0x99, 0xc9, 0x00)
        chaos_show_trigger.write_script()

    #for meo in MapEnemyObject.every:
    #    meo.cave_sanitize_events()
    f = open(path.join(tblpath, "problematic_scripts.txt"))
    for line in f:
        pointer = int(line.strip(), 0x10)
        s = Script.get_by_pointer(pointer)
        s.remove_exit_mouse_store()
        s.remove_encounters_off()
        s.remove_teleports()
        s.remove_party_changes()
        s.fix_hotels()
    f.close()
    for s in Script._all_scripts:
        s.fulfill_scheduled_write()


def replace_sanctuary_bosses():
    sbosses = [MapSpriteObject.get(sbi) for sbi in SANCTUARY_BOSS_INDEXES]
    sclusters = [mso.nearest_cluster for mso in sbosses]
    sclusters = sorted(sclusters, key=lambda sc: sc.rank)

    print "Finding bosses..."
    bosses = set([])
    for mso in MapSpriteObject.every:
        if mso.script is None:
            continue
        es = mso.script.enemy_encounters
        for e in es:
            bosses.add(e)

    bosses = sorted(bosses,
                    key=lambda b: (b.rank, random.random(), b.index))

    BANNED = [0x1ce, 0x1c8]  # clumsy robot and master belch
    bosses = [b for b in bosses if b.index not in BANNED]

    chosens = []
    while len(chosens) < len(sclusters):
        candidates = [b for b in bosses if b not in chosens]
        max_index = len(candidates)-1
        index = random.randint(random.randint(0, max_index), max_index)
        chosens.append(candidates[index])

    bosses = shuffle_normal(bosses)
    chosens = sorted(chosens, key=lambda c: bosses.index(c))

    done_scripts = set([])
    for i, (sc, c) in enumerate(zip(sclusters, chosens)):
        numbered_boss = sbosses[i]
        sboss = [s for s in sbosses if s.nearest_cluster is sc][0]
        old_script = sboss.script
        sboss.set_script(numbered_boss)
        if sboss is not numbered_boss:
            assert sboss.script is not old_script
        else:
            assert sboss.script is old_script
        sboss.script.replace_sanctuary_boss(c)
        assert sboss.script not in done_scripts
        done_scripts.add(sboss.script)
    assert len(done_scripts) == 8


class EnemyPlaceObject(TableObject):
    @classproperty
    def after_order(self):
        return [BattleEntryObject]

    def __repr__(self):
        s = ""
        for i, rate in enumerate(self.sub_group_rates):
            if rate == 0:
                continue
            s += "\nENEMY PLACEMENT %x-%s %s/100" % (self.index, "ab"[i], rate)
            for prob, beo in zip(self.odds[i], self.battle_entries[i]):
                s += "\n  %s %s" % (prob, str(beo).replace("\n", "\n    "))
        return s.strip()
    
    def serialize(self):
        if self.index == 0:
            return None
        result = {
            "index": self.index,
            "flag": self.event_flag,
            "subgroups": []
        }
        for i, rate in enumerate(self.sub_group_rates):
            if rate == 0:
                continue
            subgroup = {
                "subgroup": i,
                "rate": rate,
                "entries": []
            }
            for prob, beo in zip(self.odds[i], self.battle_entries[i]):
                subgroup["entries"].append({
                    "probability": prob,
                    "enemyEncounter": beo
                })
            result["subgroups"].append(subgroup)
        return result


    def read_data(self, filename, pointer=None):
        super(EnemyPlaceObject, self).read_data(filename, pointer)
        f = open(filename, "r+b")
        pointer = self.placement_group_pointer
        assert (pointer & 0xC00000) == 0xC00000
        pointer = pointer & 0x3FFFFF
        f.seek(pointer)
        self.event_flag = read_multi(f, length=2)
        self.sub_group_rates = map(ord, f.read(2))
        self.odds = defaultdict(list)
        self.battle_entries = defaultdict(list)
        for i, rate in enumerate(self.sub_group_rates):
            if rate == 0:
                continue
            while True:
                prob = ord(f.read(1))
                battle_entry = read_multi(f, length=2)
                self.odds[i].append(prob)
                self.battle_entries[i].append(
                        BattleEntryObject.get(battle_entry))
                if sum(self.odds[i]) == 8:
                    break
        f.close()

    @cached_property
    def rank(self):
        if self.index == 0:
            return -1

        if 0 not in self.sub_group_rates or self.sub_group_rates[0] == 0:
            return -1

        #if sum(self.sub_group_rates.values()) == 0:
        #    return -1

        try:
            return max([beo.rank for i in self.battle_entries.keys()
                        for beo in self.battle_entries[i]])
        except ValueError:
            return -1

    @classproperty
    def valid_ranked_placements(cls):
        if hasattr(EnemyPlaceObject, "_valid_ranked_placements"):
            return EnemyPlaceObject._valid_ranked_placements

        EnemyPlaceObject._valid_ranked_placements = [
            epo for epo in EnemyPlaceObject.ranked if epo.rank > 0]
        return EnemyPlaceObject.valid_ranked_placements


class BattleEntryObject(TableObject):
    @classproperty
    def after_order(self):
        return [EnemyObject]

    def __repr__(self):
        s = "BATTLE ENTRY %x" % self.index
        for a, e in zip(self.activities, self.enemies):
            s += "\n{0:0>2} {1}".format("%x" % a, e)
        return s.strip()

    def serialize(self):
        result = []
        for a, e in zip(self.activities, self.enemies):
            result.append({ "activity": a, "enemy": e.name })
        return result

    def read_data(self, filename, pointer=None):
        super(BattleEntryObject, self).read_data(filename, pointer)
        f = open(filename, "r+b")
        pointer = self.enemies_pointer
        assert (pointer & 0xC00000) == 0xC00000
        pointer = pointer & 0x3FFFFF
        f.seek(pointer)
        activities = []
        enemies = []
        while True:
            enemy_active = ord(f.read(1))
            if enemy_active == 0xFF:
                break
            enemy_id = read_multi(f, length=2)
            e = EnemyObject.get(enemy_id)
            enemies.append(e)
            activities.append(enemy_active)
        self.enemies = enemies
        self.activities = activities
        f.close()

    @cached_property
    def rank(self):
        return max([e.rank for e in self.enemies])


class ItemObject(TableObject):
    @property
    def name(self):
        return bytes_to_text(self.name_text)

    @cached_property
    def is_buyable(self):
        for s in ShopObject.every:
            if self.index in s.old_data["item_ids"]:
                return True
        return False

    @property
    def is_equipment(self):
        return self.item_type in [0x10, 0x11, 0x14, 0x18, 0x1c]

    @property
    def is_sellable(self):
        return self.old_data["price"] > 0

    @property
    def is_key_item(self):
        return (self.item_type in [0, 0x34, 0x35, 0x38, 0x3a, 0x3b]
                and not (self.is_buyable or self.is_sellable))

    @property
    def rank(self):
        if self.index == 0: # 'Null' item has a cost so will get incorrectly ranked
            return -1

        if self.is_key_item:
            if 'a' in get_flags() and not self.get_bit("nogive"):
                return 1000001
            return -1

        if not self.is_buyable and not self.is_sellable:
            return 1000000

        if not self.is_buyable:
            return 100000 + self.old_data["price"]

        return self.old_data["price"]

    @property
    def limit_one(self):
        if self.get_bit("one_use"):
            return False

        if self.price == 0 or self.get_bit("nogive"):
            return True

        if self.equipable & 0xF:
            equipnum = sum([self.get_bit(name)
                            for name in ["ness", "paula", "jeff", "poo"]])
            assert equipnum > 0
            if equipnum <= 1:
                return True

        return False
    
    @property
    def script_sources(self):
        return [s for s in Script.every if self in s.items_given]

    @property
    def chest_sources(self):
        return [t for t in MapSpriteObject.every if t.is_chest and self == t.chest_contents]
    
    @property
    def all_sources(self):
        sources = self.script_sources
        sources.extend(self.chest_sources)
        return sources

    def cleanup(self):
        if 'a' in get_flags() and not (
                self.is_sellable or self.get_bit("nogive")):
            self.price = max(self.price, 2)
        
        if "devmode" in get_activated_codes() and self.index == 0x9e: # spawn toggler
            self.name_text = text_to_bytes("Spawn Toggler", 25)
            toggle_script = Script.get_by_pointer(0x6fc94)
            toggle_script.lines = [
                (0x01,),
                (0x06, 0x0b, 0x00, 0xaf, 0xfc, 0xc6, 0x00),
                text_to_values("@Spawns set off", False),
                (0x04, 0x0b, 0x00),
                (0x13,),
                (0x02,),
                text_to_values("@Spawns set on", False),
                (0x05, 0x0b, 0x00),
                (0x13,),
                (0x02,),
            ]
            toggle_script.write_script()


class ShopObject(TableObject):
    flag = 's'
    flag_description = "shops"

    @property
    def items(self):
        return [ItemObject.get(i) for i in self.item_ids if i]

    @property
    def sister_shops(self):
        sisters = []
        for i in self.item_ids:
            if not i:
                continue
            sisters.extend([s for s in ShopObject.every if i in s.item_ids])
        return sorted(sisters)

    @property
    def sister_wares(self):
        wares = []
        for s in self.sister_shops:
            wares.extend([i for i in s.item_ids if i])
        return wares

    @cached_property
    def rank(self):
        prices = [ItemObject.get(i).old_data["price"]
                  for i in self.old_data["item_ids"] if i]
        return max(prices)

    def __repr__(self):
        s = "SHOP %x" % self.index
        for i in self.items:
            s += "\n{0:25} {1: >5}".format(i.name, i.price)
        return s.strip()

    def mutate(self):
        wares = sorted(self.sister_wares)
        new_item_ids = []
        while len(new_item_ids) < 7 and len(wares) > 0:
            chosen = random.choice(wares)
            new_item_ids.append(chosen)
            wares = [w for w in wares if w != chosen]
        new_item_ids = sorted(new_item_ids)
        assert 0 not in new_item_ids
        while len(new_item_ids) < 7:
            new_item_ids.append(0)
        self.item_ids = new_item_ids


class BgDataObject(TableObject):
    flag = 'b'
    flag_description = "backgrounds"

    def mutate(self):
        matching_depths = [b for b in BgDataObject.every if b.color_depth == self.color_depth]
        source = random.choice(matching_depths)
        self.palette = source.old_data["palette"]
        self.palette_cycle = source.old_data["palette_cycle"]
        self.palette_cycle_1_begin = source.old_data["palette_cycle_1_begin"]
        self.palette_cycle_1_end = source.old_data["palette_cycle_1_end"]
        self.palette_cycle_2_begin = source.old_data["palette_cycle_2_begin"]
        self.palette_cycle_2_end = source.old_data["palette_cycle_2_end"]
        self.palette_changing_speed = source.old_data["palette_changing_speed"]

        source = random.choice(BgDataObject.every)
        self.scrolling_movement_1 = source.old_data["scrolling_movement_1"]
        self.scrolling_movement_2 = source.old_data["scrolling_movement_2"]
        self.scrolling_movement_3 = source.old_data["scrolling_movement_3"]
        self.scrolling_movement_4 = source.old_data["scrolling_movement_4"]

        source = random.choice(BgDataObject.every)
        self.distortion_1 = source.old_data["distortion_1"]
        self.distortion_2 = source.old_data["distortion_2"]
        self.distortion_3 = source.old_data["distortion_3"]
        self.distortion_4 = source.old_data["distortion_4"]

class ExperienceObject(TableObject):
    def cleanup(self):
        if 'a' in get_flags():
            per_character = 100
            level = (self.index % per_character)-2
            if level < 0:
                return
            assert self.xp
            assert level < (per_character-2)
            progress = float(level) / (per_character-3)
            assert progress <= 1.0
            progress *= 2
            if progress >= 1.0:
                return
            progress = progress ** 0.5
            assert self.xp == self.old_data['xp']
            self.xp = int(round(self.xp * progress))
            self.xp = max(self.xp, 1)
            previous = ExperienceObject.get(self.index-1)
            assert previous.xp == 0 or previous.old_data['xp'] != previous.xp
            self.xp = max(self.xp, previous.xp+1)


class EnemyObject(TableObject):
    flag = 'm'
    flag_description = "enemy stats"

    mutate_attributes = {
        "hp": None,
        "pp": None,
        "xp": None,
        "money": None,
        "level": None,
        "offense": None,
        "defense": None,
        "speed": None,
        "guts": None,
        "iq": None,
        "miss_rate": None,
        "drop_frequency": None,
        "drop_item_index": ItemObject,
        "mirror_success_rate": None,
        #"max_call": None,
        "weakness_fire": None,
        "weakness_freeze": None,
        "weakness_flash": None,
        "weakness_paralysis": None,
        "weakness_hypnosis": None,
        "battle_palette": None,
    }
    intershuffle_attributes = [
        "hp", "xp", "money", "level",
        "offense", "defense", "speed", "guts", "iq", "miss_rate",
        ("drop_item_index", "drop_frequency"), "status",
        "mirror_success_rate",
        "battle_palette",
        ]
    randomize_attributes = [
        #"order",
        ]
    shuffle_attributes = [
        #("action1", "action2", "action3", "action4"),
        ]

    def __repr__(self):
        s = "%s %x %s" % (self.__class__.__name__, self.index, self.name)
        return s.strip()

    @property
    def is_boss(self):
        return self.boss_flag or self.death_sound

    @property
    def intershuffle_valid(self):
        return not self.is_boss

    @property
    def name(self):
        return bytes_to_text(self.name_text)

    @property
    def rank(self):
        if hasattr(self, "_rank"):
            return self._rank
        by_hp = sorted(EnemyObject.every, key=lambda e: (
            e.old_data["hp"], e.old_data["xp"], e.index))
        by_xp = sorted(EnemyObject.every, key=lambda e: (
            e.old_data["xp"], e.old_data["hp"], e.index))
        for e in EnemyObject.every:
            index = max(by_hp.index(e), by_xp.index(e))
            e._rank = index
        return self.rank

    def cleanup(self):
        if self.is_boss:
            for attr in [
                    "hp", "pp", "level", "offense", "defense", "speed", "guts",
                    "iq", "weakness_fire", "weakness_freeze", "weakness_flash",
                    "weakness_paralysis", "weakness_hypnosis"]:
                setattr(self, attr,
                        max(getattr(self, attr), self.old_data[attr]))

        if 'a' in get_flags():
            self.xp = max(self.xp, 4)

        if 'easymodo' in get_activated_codes():
            self.hp = 1
            self.pp = 1
            self.offense = 1
            self.speed = 1
            self.iq = 1


class StatGrowthObject(TableObject):
    flag = 'c'
    flag_description = "character stats"

    randomize_attributes = [
        "offense", "defense", "speed", "guts", "vitality", "iq", "luck"]
    mutate_attributes = {
        "offense": None,
        "defense": None,
        "speed": None,
        "guts": None,
        "vitality": None,
        "iq": None,
        "luck": None,
        }


class InitialStatsObject(TableObject):
    def __repr__(self):
        s = "%x %s %s %s %x" % (self.index, self.level, self.xp, self.money,
                                self.unknown)
        for i in self.items:
            s += "\n%s" % i
        return s.strip()

    @property
    def items(self):
        items = []
        for i in self.item_indexes:
            if i:
                items.append(ItemObject.get(i))
        return items

    def clear_inventory(self):
        self.item_indexes = [0] * len(self.item_indexes)

    def add_item(self, item):
        if isinstance(item, ItemObject):
            item = item.index

        self.item_indexes = [i for i in self.item_indexes if i > 0]
        self.item_indexes.append(item)
        self.item_indexes = sorted(self.item_indexes, reverse=True)
        while len(self.item_indexes) < 10:
            self.item_indexes.append(0)

    def cleanup(self):
        if 'a' in get_flags():
            self.level = 1
            self.xp = 0
            if self.index == 0:
                self.money = 100
                self.add_item(0xC4)  # sound stone
                self.add_item(0x11)  # cracked bat
                self.add_item(0xC5)  # exit mouse

        if "devmode" in get_activated_codes() and self.index == 0:
            self.add_item(0x9E)  # spawn toggler

        if "easymodo" in get_activated_codes():
            self.level = 99
            self.add_item(0x01)  # franklin badge
            self.add_item(0x3E)  # star pendant
            if self.index == 0:
                self.money = 65000


class PsiAbilityObject(TableObject):
    def cleanup(self):
        if 'k' in get_flags() and 'a' not in get_flags() and self.name_index == 0x11 and self.greek_letter == 0x01: # Teleport Alpha
            self.ness_level = 1

class PsiTeleportObject(TableObject):
    flag = 'k'
    flag_description = 'with keysanity mode'
    _results = None

    @property
    def name(self):
        return bytes_to_text(self.name_text)
    
    def mutate(self):
        if 'a' in get_flags():
            return # Disable Keysanity if Ancient Cave on
        if len(self.name) > 0:
            self.flag = 0xd1 # Onett discovered
        if self.index == 13: # Add South Winters teleport
            self.name_text = text_to_bytes("South Winters", 25)
            self.x = 26
            self.y = 595
            self.flag = 0xd1 # Onett discovered
        if self.index == 15: # Add North Onett teleport
            self.name_text = text_to_bytes("North Onett", 25)
            self.x = 322
            self.y = 54
            self.flag = 0xd1 # Onett discovered

            
    @classmethod
    def check_legal_keysanity(cls):
        # Rather than a post-check, this would be better served as placing things logically in the first place.
        # But this is sufficient for now.
        if PsiTeleportObject._results is None:
            return False
        mapping = dict(PsiTeleportObject._results)
        if mapping[ItemObject.get(0xb4)] is ItemObject.get(0x01): # Franklin badge at Wad of bills
            return False
        if mapping[ItemObject.get(0x01)] is ItemObject.get(0xb8): # Pencil eraser at Franklin badge
            return False
        if mapping[ItemObject.get(0xb4)] is ItemObject.get(0xb8): # Pencil eraser at Wad of bills
            return False
        if mapping[ItemObject.get(0xfd)] is ItemObject.get(0xb7): # Signed banana at Carrot key
            return False
        if mapping[ItemObject.get(0xfd)] is ItemObject.get(0xb6): # Diamond at Carrot key
            return False
        if mapping[ItemObject.get(0xa6)] is ItemObject.get(0xb6): # Diamond at King banana
            return False
        if mapping[ItemObject.get(0xb7)] is ItemObject.get(0xb6): # Diamond at Signed banana
            return False
        if mapping[ItemObject.get(0xd3)] is ItemObject.get(0xa4): # Shyness book at Tendakraut
            return False
        return True
        

    @classmethod
    def serialize(cls):
        def serialize_result(result):
            return {
                "item": result[1].name,
                "destination": result[0].name
            }
        return map(serialize_result, PsiTeleportObject._results)

    @classmethod
    def intershuffle(cls):
        cls.class_reseed("inter")
        if 'a' in get_flags():
            return # Disable Keysanity if Ancient Cave on
        key_items_index = [
            0x01,   # Franklin badge
            #0x69,   # Jar of Fly Honey - Chest handled differently, at 0x7dacb.
            0xa4,   # Shyness book
            0xa6,   # King banana
            0xaa,   # Key to the shack
            0xaf,   # Hawk eye
            0xb0,   # Bicycle
            0xb4,   # Wad of bills
            0xb6,   # Diamond
            0xb7,   # Signed banana
            0xb8,   # Pencil eraser
            0xc0,   # Key to the tower
            0xca,   # Town map
            0xcc,   # Suporma
            0xd3,   # Tendakraut
            0xfd,   # Carrot key
        ]
        # TODO: Cache this when final key items list decided.
        a =  [me for me in MapEventObject.every]
        b = [ms for ms in MapSpriteObject.every]
        for o in a + b:
            o.script
        Script.get_by_pointer(0x9d95e) # Tendakraut

        source_items = map(lambda x: ItemObject.get(x), key_items_index)
        destination_sets = map(lambda x: x.all_sources, source_items)
        new_items = list(source_items)

        new_items.remove(ItemObject.get(0xd3)) # Tendakraut
        new_items.append(ItemObject.get(0x69)) # Jar of Fly Honey
        new_items.remove(ItemObject.get(0xcc)) # Suporma
        new_items.append(ItemObject.get(0xc1)) # Meteorite piece

        while not cls.check_legal_keysanity():
            random.shuffle(new_items)
            PsiTeleportObject._results = zip(source_items, new_items)

        for (source_item, new_item, destination_set) in zip(source_items, new_items, destination_sets):
            for destination in destination_set:
                destination.replace_item(source_item, new_item)
                destination.mutated = True


    @classmethod
    def full_cleanup(cls):
        if 'a' in get_flags():
            print "WARNING: Keysanity and Ancient Cave modes are incompatible. Keysanity has been disabled."
            super(PsiTeleportObject, cls).full_cleanup()
            return

        # Patch Bubble Monkey rope interaction
        bubble_monkey_rope = Script.get_by_pointer(0x97f72)
        lines = bubble_monkey_rope.lines
        bubble_monkey_rope.lines = lines[:1] + lines[-2:]
        bubble_monkey_rope.write_script()

        # Patch intro script to set Onett flag immediately
        intro = Script.get_by_pointer(0x5e70b)
        patch_lines = intro.lines[:2] + [
            (0x04, 0xd1, 0x00), # Enable Onett teleport
            (0x04, 0xd9, 0x00), # Enable Pyramid entrance
            (0x04, 0x8c, 0x00), # Enable Venus giving item
            (0x02, )]
        patch = Script.write_new_script(patch_lines)
        assert patch.length == 16
        intro.lines = [ccode_call_address(patch.pointer)] + [(0x00, )] + intro.lines[2:]
        intro.write_script()

        # Patch Montague to always show up
        montague = TPTObject.get(0x2f8)
        assert montague.address == 0xc60349
        montague.flag = 0

        # Patch Mr Spoon to request autograph even after he's received it
        spoon = TPTObject.get(0x38d)
        #assert spoon.address == 0xc826bc - could be changed in Dialog shuffle
        spoon.address = 0xc82468

        # Patch Bubble Monkey to appear at north shore as soon as he runs off with his gal
        monkey = Script.get_by_pointer(0x882bd)
        patch_lines = monkey.lines[:2] + [(0x04, 0x76, 0x02), (0x02, )] # Enable Monkey at north shore
        patch = Script.write_new_script(patch_lines)
        assert patch.length == 9
        monkey.lines = [ccode_call_address(patch.pointer)] + monkey.lines[2:]
        monkey.write_script()

        # Patch Dr Andonuts to recognize Ness isn't Jeff
        andonuts = Script.get_by_pointer(0x6b18d)
        patch_lines = andonuts.lines[:2] + [ # check the normal flags first
            (0x19, 0x10, 0x01), # check character in slot 1
            (0x0b, 0x03), # is it Jeff?
            (0x1b, 0x03) + ccode_address(0x6b18d), # go to normal andonuts text
            ccode_goto_address(0x6b56e), # go to generic text for Ness
            (0x02, )] 
        patch = Script.write_new_script(patch_lines)
        andonuts_tpt = TPTObject.get(0x267)
        assert andonuts_tpt.address == 0xc6b18d
        andonuts_tpt.address = 0xc00000 + patch.pointer

        super(PsiTeleportObject, cls).full_cleanup()
     

if __name__ == "__main__":
    try:
        print ("You are using the Earthbound Ancient Cave "
               "randomizer version %s." % VERSION)
        print

        ALL_OBJECTS = [g for g in globals().values()
                       if isinstance(g, type) and issubclass(g, TableObject)
                       and g not in [TableObject]]

        codes = {
            "easymodo": ["easymodo"],
            "mapper": ["mapper"],
            "giygastest": ["giygastest"],
            "funsize": ["funsize"],
            "wildgifts": ["wildgifts"],
            "devmode": ["devmode"]
        }
        run_interface(ALL_OBJECTS, snes=True, codes=codes)

        clean_and_write(ALL_OBJECTS)
        rewrite_snes_meta("EB-AC", VERSION, lorom=False)

        # Spoiler / Map data generation
        print("Making spoiler/map file...")
        spoiler_file = open((get_outfile()[:-4] + ".spoiler.json"), "w")
        spoiler_object = {
            "info": {
                "version": VERSION,
                "seed": get_seed(),
                "flags": get_flags(),
                "codes": get_activated_codes(),
                "timestamp": int(time() * 1000)
            },
            "chests": [m for m in MapSpriteObject.every if m.is_chest]
        }
        if 'k' in get_flags() and 'a' not in get_flags():
            spoiler_object["keysanity"] = PsiTeleportObject.serialize()
        if 'a' in get_flags():
            Cluster.mark_shortest_path()
        if 'a' in get_flags() or "devmode" in get_activated_codes():
            spoiler_object["clusters"] = Cluster.generate_clusters()
            spoiler_object["bosses"] = [mso for mso in MapSpriteObject.every if mso.index in SANCTUARY_BOSS_INDEXES]
        if "devmode" in get_activated_codes():
            # This adds multiple MB to the file and is only useful for developers, so do not generate it by default.
            spoiler_object["enemies"] = MapEnemyObject.every
        json.dump(spoiler_object, spoiler_file, default=(lambda x: x.serialize()))
        spoiler_file.close()

        if "mapper" in get_activated_codes():
            Cluster.generate_map()

        finish_interface()

    except Exception, e:
        print "ERROR: %s" % e
        raw_input("Press Enter to close this program.")
