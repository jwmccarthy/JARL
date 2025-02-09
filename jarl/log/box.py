from dataclasses import dataclass


@dataclass
class LightArcBox:
    TOP_LEFT    = "╭"
    TOP_RIGHT   = "╮"
    BOT_LEFT    = "╰"
    BOT_RIGHT   = "╯"
    HORIZONTAL  = "─"
    VERTICAL    = "│"
    INTERSECT   = "┼"
    TOP_INTER   = "┬"
    BOT_INTER   = "┴"
    LEFT_INTER  = "├"
    RIGHT_INTER = "┤"


@dataclass
class DoubleBox:
    TOP_LEFT    = "╔"
    TOP_RIGHT   = "╗"
    BOT_LEFT    = "╚"
    BOT_RIGHT   = "╝"
    HORIZONTAL  = "═"
    VERTICAL    = "║"
    INTERSECT   = "╬"
    TOP_INTER   = "╦"
    BOT_INTER   = "╩"
    LEFT_INTER  = "╠"
    RIGHT_INTER = "╣"


@dataclass
class SingleBox:
    TOP_LEFT    = "┌"
    TOP_RIGHT   = "┐"
    BOT_LEFT    = "└"
    BOT_RIGHT   = "┘"
    HORIZONTAL  = "─"
    VERTICAL    = "│"
    INTERSECT   = "┼"
    TOP_INTER   = "┬"
    BOT_INTER   = "┴"
    LEFT_INTER  = "├"
    RIGHT_INTER = "┤"