import uuid
from copy import copy
from typing import Optional
import numpy as np
import modules.common as c
import modules.memory as m
import modules.queue as q
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s|%(filename)s|%(lineno)s| %(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    filename='organism.log',
)
logger = logging.getLogger(__name__)


class RegsDict(dict):
    allowed_keys = ['a', 'b', 'c', 'd']

    def __setitem__(self, key, value):
        if key not in self.allowed_keys:
            raise ValueError
        super().__setitem__(key, value)


class Organism:
    def __init__(
            self,
            address: np.array,
            size: np.array,
            ip: Optional[np.array] = None,
            delta: Optional[np.array] = np.array([0, 1]),
            start: Optional[np.array] = None,
            regs: Optional[RegsDict] = None,
            stack: Optional[list] = None,
            errors: Optional[int] = 0,
            child_size: Optional[np.array] = np.array([0, 0]),
            child_start: Optional[np.array] = np.array([0, 0]),
            is_selected: Optional[bool] = False,
            children: Optional[int] = 0,
            reproduction_cycle: Optional[int] = 0,
            parent: Optional[uuid.UUID] = None,
            organism_id: Optional[uuid.UUID] = None,
            organism_padding: Optional[np.array] = np.array([1, 1]),
    ):
        # pylint: disable=invalid-name
        self.organism_id = uuid.uuid4() if organism_id is None else organism_id
        self.organism_padding = organism_padding
        self.parent = parent
        # pylint: disable=invalid-name
        self.ip = np.array(address) if ip is None and address is not None else ip
        self.delta = delta
        self.center_flag = False

        self.size = np.array(size)
        self.start = (
            np.array(address) if start is None and address is not None else start
        )
        self.regs = (
            RegsDict(
                {
                    'a': np.array([0, 0]),
                    'b': np.array([0, 0]),
                    'c': np.array([0, 0]),
                    'd': np.array([0, 0]),
                }
            )
            if regs is None
            else regs
        )
        self.stack = [] if stack is None else stack

        self.errors = errors

        self.child_size = child_size
        self.child_start = child_start

        self.is_selected = is_selected

        if address is not None:
            m.memory.allocate(address, size)

        self.reproduction_cycle = reproduction_cycle
        self.children = children

        q.queue.add_organism(self)
        if start is None and address is not None:
            q.queue.archive.append(copy(self))

        self.mods = {'x': 0, 'y': 1}

    def no_operation(self):
        pass

    def move_up(self):
        self.delta = c.deltas['up'] * abs(max(self.delta.min(), self.delta.max(), key=abs))

    def move_down(self):
        self.delta = c.deltas['down'] * abs(max(self.delta.min(), self.delta.max(), key=abs))

    def move_right(self):
        self.delta = c.deltas['right'] * abs(max(self.delta.min(), self.delta.max(), key=abs))

    def move_left(self):
        self.delta = c.deltas['left'] * abs(max(self.delta.min(), self.delta.max(), key=abs))

    def ip_offset(self, offset: int = 0) -> np.array:
        return self.ip + offset * self.delta

    def inst(self, offset: int = 0) -> str:
        return m.memory.inst(self.ip_offset(offset))

    def find_template(self):
        register = self.inst(1)
        template = []
        for i in range(2, max(self.size)):
            if self.inst(i) in ['.', ':']:
                template.append(':' if self.inst(i) == '.' else '.')
            else:
                break
        counter = 0
        for i in range(i, max(self.size)):
            if self.inst(i) == template[counter]:
                counter += 1
            else:
                counter = 0
            if counter == len(template):
                self.regs[register] = self.ip + i * self.delta
                break

    def if_not_zero(self):
        if self.inst(1) in self.mods.keys():
            value = self.regs[self.inst(2)][self.mods[self.inst(1)]]
            start_from = 1
        else:
            value = self.regs[self.inst(1)]
            start_from = 0

        if not np.any(value):
            self.ip = self.ip_offset(start_from + 1)
        else:
            self.ip = self.ip_offset(start_from + 2)

    def increment(self):
        if self.inst(1) in self.mods.keys():
            self.regs[self.inst(2)][self.mods[self.inst(1)]] += 1
        else:
            self.regs[self.inst(1)] += 1

    def decrement(self):
        if self.inst(1) in self.mods.keys():
            self.regs[self.inst(2)][self.mods[self.inst(1)]] -= 1
        else:
            self.regs[self.inst(1)] -= 1

    def zero(self):
        self.regs[self.inst(1)] = np.array([0, 0])

    def one(self):
        self.regs[self.inst(1)] = np.array([1, 1])

    def subtract(self):
        self.regs[self.inst(3)] = self.regs[self.inst(1)] - self.regs[self.inst(2)]

    def allocate_child(self):

        """
        Firstly, let\'s get new child size. In order to properly do it, we need to make x3 of the whole size, because
        each command cell is now 3x3
        """
        size = np.copy(self.regs[self.inst(1)]) * 3

        if (size <= 0).any():
            return
        is_space_found = False

        """
        Then, we need to seek, BUT!
        We need to move one by one => either we need to change ip_offset, or we need to change delta. 
        Changing delta is a simpler one, because it doesn\'t require rewriting the whole algorithm.  
        """
        delta_before = np.copy(self.delta)

        # Diving by absolute makes retain direction, but in a way like [1, -1] or [0, -1]
        nonzero_in_delta = self.delta != 0

        self.delta[nonzero_in_delta] = self.delta[nonzero_in_delta] / np.absolute(self.delta[nonzero_in_delta])
        print(size)

        for i in range(2, max(c.config['memory_size'])):
            is_allocated_region = m.memory.is_allocated_region(self.ip_offset(i), size)

            if is_allocated_region is None:
                break
            if not is_allocated_region:
                self.child_start = self.ip_offset(i)
                self.regs[m.memory.inst(self.ip + delta_before * 2)] = np.copy(self.child_start)
                is_space_found = True
                break

        if is_space_found:
            self.child_size = np.copy(size // 3)
            m.memory.allocate(self.child_start, self.child_size)
        print(self.delta)
        self.delta = delta_before

    def load_inst(self):
        self.regs[self.inst(2)] = c.instructions[
            m.memory.inst(self.regs[self.inst(1)])
        ][0]

    def write_inst(self):
        if not np.array_equal(self.child_size, np.array([0, 0])):
            m.memory.write_inst(self.regs[self.inst(1)], self.regs[self.inst(2)])

    def push(self):
        if len(self.stack) < c.config['stack_length']:
            self.stack.append(np.copy(self.regs[self.inst(1)]))

    def correct_error(self):
        next = self.delta // max(np.abs(self.delta)) + self.ip
        center = next
        coords_for_voting = [center]

        for x_offset in [-1, 1]:
            for y_offset in [-1, 1]:
                coords_for_voting.append(np.array([center[0] + x_offset, center[1] + y_offset]))

        votes = []
        for voting_coord in coords_for_voting:
            votes.append(m.memory.inst(voting_coord))
        mode = max(set(votes), key=votes.count)
        for voting_coord in coords_for_voting:
            m.memory.write_inst(voting_coord, c.instructions[mode][0])

    def pop(self):
        self.regs[self.inst(1)] = np.copy(self.stack.pop())

    def ip_is_center(self):
        center = self.ip
        votes = []
        coords_for_voting = []
        for x_offset in [-1, 1]:
            coords_for_voting.append(np.array([center[0] + x_offset, center[1]]))

        for y_offset in [-1, 1]:
            coords_for_voting.append(np.array([center[0], center[1] + y_offset]))

        for voting_coord in coords_for_voting:
            votes.append(m.memory.inst(voting_coord))
        value_around = set(votes)

        return len(value_around) == 1 and value_around.pop() == 'E'

    def split_child(self):
        if not np.array_equal(self.child_size, np.array([0, 0])):
            m.memory.deallocate(self.child_start, self.child_size * 3)
            self.__class__(self.child_start, self.child_size * 3, parent=self.organism_id)
            self.children += 1
            self.reproduction_cycle = 0
        self.child_size = np.array([0, 0])
        self.child_start = np.array([0, 0])

    def __lt__(self, other):
        return self.errors < other.errors

    def kill(self):
        m.memory.deallocate(self.start, self.size)
        self.size = np.array([0, 0])
        if not np.array_equal(self.child_size, np.array([0, 0])):
            m.memory.deallocate(self.child_start, self.child_size)
        self.child_size = np.array([0, 0])

    def cycle(self):
        is_center = self.ip_is_center()
        if is_center:
            self.delta *= 3
        try:
            getattr(self, c.instructions[self.inst()][1])()
            if (
                    c.config['penalize_parasitism']
                    and not m.memory.is_allocated(self.ip)
                    and max(np.abs(self.ip - self.start)) > c.config['penalize_parasitism']
            ):
                raise ValueError
        except Exception as e:
            # print(str(e))
            self.errors += 1
        if is_center:
            self.delta = self.delta // 3
        if is_center:
            self.delta *= 2
            self.center_flag = True
        else:
            self.delta = self.delta // 2 if self.center_flag else self.delta
            self.center_flag = False

        new_ip = self.ip + self.delta
        self.reproduction_cycle += 1
        if (
                self.errors > c.config['organism_death_rate']
                or self.reproduction_cycle > c.config['kill_if_no_child']
        ):
            q.queue.organisms.remove(self)
            self.kill()
        if (new_ip < 0).any() or (new_ip - c.config['memory_size'] > 0).any():
            return None
        self.ip = np.copy(new_ip)

        return None

    def update(self):
        pass

    def toogle(self):
        OrganismFull(
            address=None,
            size=self.size,
            ip=self.ip,
            delta=self.delta,
            start=self.start,
            regs=self.regs,
            stack=self.stack,
            errors=self.errors,
            child_size=self.child_size,
            child_start=self.child_start,
            is_selected=self.is_selected,
            children=self.children,
            reproduction_cycle=self.reproduction_cycle,
            parent=self.parent,
            organism_id=self.organism_id,
        )


class OrganismFull(Organism):
    def __init__(
            self,
            address: np.array,
            size: np.array,
            ip: Optional[np.array] = None,
            delta: Optional[np.array] = np.array([0, 1]),
            start: Optional[np.array] = None,
            regs: Optional[RegsDict] = None,
            stack: Optional[list] = None,
            errors: Optional[int] = 0,
            child_size: Optional[np.array] = np.array([0, 0]),
            child_start: Optional[np.array] = np.array([0, 0]),
            is_selected: Optional[bool] = False,
            children: Optional[int] = 0,
            reproduction_cycle: Optional[int] = 0,
            parent: Optional[uuid.UUID] = None,
            organism_id: Optional[uuid.UUID] = None,
    ):
        super(OrganismFull, self).__init__(
            address=address,
            size=size,
            ip=ip,
            delta=delta,
            start=start,
            regs=regs,
            stack=stack,
            errors=errors,
            child_size=child_size,
            child_start=child_start,
            is_selected=is_selected,
            children=children,
            reproduction_cycle=reproduction_cycle,
            parent=parent,
            organism_id=organism_id,
        )

        self.update()

    def update_window(self, size, start, color):
        new_start = start - m.memory.position
        new_size = size + new_start.clip(max=0)

        if (new_size > 0).all() and (m.memory.size - new_start > 0).all():
            m.memory.window.derived(
                new_start.clip(min=0),
                np.amin([new_size, m.memory.size - new_start, m.memory.size], axis=0),
            ).background(color)

    def update_window_with_ip(self, size, start, color):
        self.update_window(size, start, 0)
        self.update_window(np.array([1, 1]), self.ip, color)

    def update_ip(self):
        new_position = self.ip - m.memory.position
        color = c.colors['ip_bold'] if self.is_selected else c.colors['ip']
        if (
                (new_position >= 0).all()
                and (m.memory.size - new_position > 0).all()
                and m.memory.is_allocated(self.ip)
        ):
            m.memory.window.derived(new_position, (1, 1,)).background(0)

    def update(self):
        parent_color = (
            c.colors['parent_bold'] if self.is_selected else c.colors['parent']
        )
        if self.is_selected:
            self.update_window_with_ip(self.size, self.start, parent_color)
        else:
            self.update_window(self.size, self.start, parent_color)
        child_color = c.colors['child_bold'] if self.is_selected else c.colors['child']
        self.update_window(self.child_size * 3, self.child_start, child_color)
        # m.memory.update(refresh=True)
        self.update_ip()

    def info(self):
        info = ''
        info += '  errors   : {}\n'.format(self.errors)
        info += '  ip       : {}\n'.format(list(self.ip))
        info += '  delta    : {}\n'.format(list(self.delta))
        info += '  center flag    : {}\n'.format(self.center_flag)

        for reg in self.regs:
            info += '  r{}       : {}\n'.format(reg, list(self.regs[reg]))
        for i in range(len(self.stack)):
            info += '  stack[{}] : {}\n'.format(i, list(self.stack[i]))
        for i in range(len(self.stack), c.config['stack_length']):
            info += '  stack[{}] : \n'.format(i)
        return info

    def kill(self):
        super(OrganismFull, self).kill()
        self.update()

    def toogle(self):
        Organism(
            address=None,
            size=self.size,
            ip=self.ip,
            delta=self.delta,
            start=self.start,
            regs=self.regs,
            stack=self.stack,
            errors=self.errors,
            child_size=self.child_size,
            child_start=self.child_start,
            is_selected=self.is_selected,
            children=self.children,
            reproduction_cycle=self.reproduction_cycle,
            parent=self.parent,
            organism_id=self.organism_id,
        )
