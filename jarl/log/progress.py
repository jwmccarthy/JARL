import cursor
from typing import Any, Dict
from colorama import Fore, Style
from collections import defaultdict


BUFFER = 16
KEYLEN = 16
PCTLEN = 9
BLOCKS = [' ', '▏', '▎', '▍', '▌', '▋', '▊', '▉', '█']


class ProgressBar:

    def __init__(
        self,
        total: int, 
        width: int = 32,
        color: str = Fore.GREEN,
        show_box: bool = True
    ) -> None:
        self.total = total
        self.width = width
        self.color = color
        self.lines = 0
        self.stats = defaultdict(dict)
        self.show_box = show_box

    def _build_bar(self, progress: float) -> str:
        width = self.width - PCTLEN

        # get full and partials
        fill = progress * width
        full = int(fill)
        part = fill - full
        
        # get partial block
        part_idx = int(round(part * (len(BLOCKS) - 1)))
        
        # construct progress bar
        bar_str = BLOCKS[-1] * full
        if full < width:
            bar_str += BLOCKS[part_idx]
            bar_str += BLOCKS[0] * (width - full - 1)
        
        return bar_str
    
    def _render_row(self, name: str, stats: Dict[str, Any]) -> str:
        row = ""

        # row header
        mid = '─' * self.width
        row += f"│{name.ljust(self.width+1)}│\n"
        row += f"├{mid[:KEYLEN]}┬{mid[KEYLEN:]}┤\n"

        # fill in row content
        for key, val in stats.items():
            key, val = key[:KEYLEN-2], f"{val:.3f}"
            pad = " " * (self.width - len(val) - KEYLEN - 1)
            row += f"│•{key:>{KEYLEN-2}} │ {val}{pad}│\n"

        return row
    
    def _render_box(self) -> str:
        mid = '─' * self.width
        box = f"╭{mid[:KEYLEN]}─{mid[KEYLEN:]}╮\n"
        div = f"├{mid[:KEYLEN]}┴{mid[KEYLEN:]}┤\n"
        bot = f"╰{mid[:KEYLEN]}┴{mid[KEYLEN:]}╯"

        for i, (key, val) in enumerate(self.stats.items()):
            box += div if i > 0 else ""
            box += self._render_row(key, val)

        self.lines = box.count('\n') + 1

        return box + bot
    
    def _render_bar(self, progress: float) -> str:
        # clear existing lines
        for _ in range(self.lines + BUFFER):
            print('\033[F\033[K', end='')

        # print new lines
        if self.show_box and self.stats:
            print(self._render_box())

        # print progress bar
        bar = self._build_bar(progress)
        bar = f"|{self.color}{bar}{Style.RESET_ALL}|"
        pct = f"[{progress * 100:6.2f}%]"
        print(f"{bar} {pct}", flush=True)

    def update(self, key: str, data: Dict[str, Any]) -> None:
        self.stats[key].update(**data)

    def close(self) -> None:
        self.stats = {}
        cursor.show()

    def __iter__(self):
        cursor.hide()
        for i in range(self.total):
            self._render_bar(i / self.total)
            self.lines = 0
            yield i
        self.close()