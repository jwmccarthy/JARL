from colorama import Fore, Style
from collections import defaultdict

from jarl.log.box import LightArcBox


BLOCKS = [" ", "▏", "▎", "▍", "▌", "▋", "▊", "▉", "█"]


class Progress:

    def __init__(
        self, 
        total: int, 
        width: int = 32,
        ratio: float = 0.65, 
        chars = LightArcBox
    ) -> None:
        self.total = total
        self.width = width
        self.ratio = ratio
        self.chars = chars
        self.lines = 1
        self.stats = defaultdict(dict)

    def _build_bar(self, prog: float) -> str:
        # percentage indicator
        pct_str = f"[{prog * 100:6.2f}%]"

        width = self.width - len(pct_str) - 2

        # get full and partials
        fill = prog * width
        full = int(fill)
        part = fill - full
        
        # get partial block
        part_idx = int(round(part * (len(BLOCKS) - 1)))
        
        # construct progress bar
        bar_str = BLOCKS[-1] * full
        if full < width:
            bar_str += BLOCKS[part_idx]
            bar_str += BLOCKS[0] * (width - full - 1)
        bar_str = f"|{Fore.GREEN}{bar_str}{Style.RESET_ALL}|"
        
        return bar_str + pct_str
    
    def _build_section(
        self, 
        section,
        int_width: int,
        key_width: int,
        val_width: int
    ) -> str:
        table_str = ""

        # section header
        header = section[:int_width]
        table_str += (
            self.chars.VERTICAL
            + header.ljust(int_width)
            + self.chars.VERTICAL + "\n"
        )

        # section divider
        table_str += (
            self.chars.LEFT_INTER
            + self.chars.HORIZONTAL * key_width
            + self.chars.TOP_INTER
            + self.chars.HORIZONTAL * val_width
            + self.chars.RIGHT_INTER + "\n"
        )

        # section content
        for key, ten in self.stats[section].items():
            # round tensors
            val_str = f"{float(ten):.3f}"

            # truncate keys if too long
            key_str = f"• {key[:key_width]}"
            key_str = key_str[:key_width]
            val_str = val_str[:val_width]

            # create row w/ content
            table_str += (
                self.chars.VERTICAL
                + key_str.ljust(key_width)
                + self.chars.VERTICAL
                + val_str.rjust(val_width)
                + self.chars.VERTICAL + "\n"
            )

        return table_str

    def _build_table(self) -> str:
        table_str = ""
        int_width = self.width - 2
        key_width = int((int_width - 1) * self.ratio)
        val_width = (int_width - 1) - key_width

        # top border
        table_str += (
            self.chars.TOP_LEFT
            + self.chars.HORIZONTAL * int_width
            + self.chars.TOP_RIGHT + "\n"
        )

        sections = list(self.stats.keys())
        for i, section in enumerate(sections):

            # format section
            table_str += self._build_section(
                section, int_width, key_width, val_width
            )

            # handle border on last iteration
            last = not (i == len(sections) - 1)
            lchar = self.chars.LEFT_INTER if last else self.chars.BOT_LEFT
            rchar = self.chars.RIGHT_INTER if last else self.chars.BOT_RIGHT

            # section divider if not last
            table_str += (
                lchar
                + self.chars.HORIZONTAL * key_width
                + self.chars.BOT_INTER
                + self.chars.HORIZONTAL * val_width
                + rchar
            ) + "\n" * last

        return table_str

    def update(self, **kwargs):
        for key, val in kwargs.items():
            self.stats[key].update(val)

    def close(self):
        self.stats = {}


    def __iter__(self):
        for t in range(self.total):
            tab = self._build_table()
            bar = self._build_bar((t + 1) / self.total)
            self.lines = tab.count('\n') + 1

            # clear existing lines
            for _ in range(self.lines * 2):
                print("\033[A\033[K", end="")

            if self.lines > 2:
                print(tab)
            print(bar, flush=True)

            yield t

        self.close()