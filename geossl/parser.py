"""
A small list of utilities to transform dataclasses from argparsers

>>> @dataparser
... class MyArguments:
...     x: int = 0
...     y: str = "optional"

>>> MyArguments()
MyArguments(x=0, y='optional')

>>> MyArguments(x=20)
MyArguments(x=20, y='optional')
"""

from typing import Any, Dict, Union, List, Optional

import argparse
import dataclasses


def _post_init(cls, *args, **kwargs):
    for k in dir(cls):
        v = getattr(cls, k)
        if isinstance(v, _Field):
            if v.has_default():
                setattr(cls, k, v.get_default())
            else:
                raise Exception(f"No value provided for cls {cls.__class__}.{k}")


def dataparser(cls=None):
    """
    A class decorator to use in place of @dataclasses.dataclass

    >>> @dataparser
    ... class A:
    ...     x: int = 0

    >>> A()
    A(x=0)
    """

    def wrapper(cls):
        setattr(cls, dataclasses._POST_INIT_NAME, _post_init)
        cls = dataclasses.dataclass(cls)
        return cls

    if cls is None:
        return wrapper

    return wrapper(cls)


class _MISSING:
    def __repr__(self):
        return "<missing>"


@dataclasses.dataclass()
class _Field:
    "Used to add metadata for positional args for."

    help: Optional[str] = None
    choices: Optional[List[Any]] = None
    positional: bool = False
    default: Any = _MISSING()
    required: Optional[bool] = None
    metavar: Optional[str] = None
    action: 'Optional[Literal["store_true"]]' = None

    def has_default(self) -> bool:
        try:
            self.get_default()
            return True
        except:
            return False

    def get_default(self):
        if not isinstance(self.default, _MISSING):
            return self.default

        if self.action == "store_true":
            return False

        if self.action == "store_false":
            return True

        if not self.positional and not self.required:
            return None

        raise(Exception("no default value specified"))


def Field(*args, **kwargs) -> Any:  # Acts as a type inference barrier for mypy
    """
    Used to add metadata for positional args for.

    >>> @dataparser
    ... class Args:
    ...     x: int = Field(positional=True, required=True)

    Parameters
    ==========
        help - Optional[str]
        choices - Optional[List[Any]] = None
        positional - bool
        default - Optional[Any]
        required - Optional[bool] = None
        metavar - Optional[str] = None
        action - Optional[Literal["store_true", "store_false"]] = None
    """
    return _Field(*args, **kwargs)


def from_args(cls):
    """
    Returns an instance of the dataclass with fields populated using the program arguments.

    Parameters
    ==========
        cls: A dataclass type
    Returns
    =======
        args: cls - An instance of cls
    """
    parser = to_argparser(cls)
    namespace = parser.parse_args()
    return cls(**namespace.__dict__)


def _is_optional(t):
    if hasattr(t, "__origin__") and t.__origin__ == Union and t.__slots__ is None:
        return True
    return False


def _get_optional_type(t):
    return t.__args__[0]


def to_argparser(cls) -> argparse.ArgumentParser:
    """
    Converts an annotated dataclass to a corresponding ArgumentParser.

    Parameters
    ==========
        cls: A dataclass type
    Returns
    =======
        parser: argparse.ArgumentParser - an argument parser
    """

    annotations = cls.__dict__["__dataclass_fields__"]
    help_str = "" if "__doc__" not in cls.__dict__ else cls.__doc__

    parser = argparse.ArgumentParser(help_str)
    for k, v in annotations.items():
        if _is_optional(v.type):
            argtype = _get_optional_type(v.type)
        else:
            argtype = v.type

        if isinstance(v.default, _Field):
            field = v.default
            kwargs: Dict[str, Any] = {}
            has_default = not isinstance(field.default, _MISSING)

            if field.has_default():
                kwargs["default"] = field.get_default()

            if field.choices is not None:
                kwargs["choices"] = field.choices

            if field.action is not None:
                parser.add_argument(f"--{k}", action=field.action)
                continue

            if not field.positional:
                required = (
                    field.required if field.required is not None else not field.has_default()
                )
                kwargs["required"] = required

            if field.help is not None:
                kwargs["help"] = field.help
            elif has_default:
                kwargs["help"] = f"{k.upper()} [default={field.default}]"

            if field.metavar is not None:
                kwargs["metavar"] = field.metavar

            prefix = "--" if not field.positional else ""

            parser.add_argument(f"{prefix}{k}", type=argtype, **kwargs)
        else:
            required = isinstance(v.default, dataclasses._MISSING_TYPE)
            kwargs = {
                "required": required,
            }

            if not required:
                kwargs["default"] = v.default
                kwargs["help"] = f"[default={v.default}]"

            parser.add_argument(f"--{k}", type=argtype, **kwargs)

    return parser


## Tests


def test_argparse():
    @dataparser
    class MyArguments:
        x: int = 0
        y: str = "optional"

    parser = to_argparser(MyArguments)
    args = parser.parse_args(["--x", "10", "--y", "y_value"])
    assert args.x == 10
    assert args.y == "y_value"


def test_optional():
    @dataparser
    class MyClass:
        x: Optional[int] = None

    parser = to_argparser(MyClass)
    args = parser.parse_args([])
    assert args.x is None

    args = parser.parse_args(["--x", "1"])
    assert args.x == 1


def test_field_mypy():
    import pytest

    @dataparser
    class MyArguments:
        x: int = Field(positional=True)

    parser = to_argparser(MyArguments)
    args = parser.parse_args(["1"])
    assert args.x == 1

    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_help():
    @dataparser
    class MyArguments:
        """
        A docstring
        """

        x: int = Field(help="A help string")

    parser = to_argparser(MyArguments)
    help = parser.format_help()

    assert "A docstring" in help
    assert "A help string" in help


def test_choices():
    @dataparser
    class MyArguments:
        x: int = Field(choices=[1,2,3], required=False)
        y: int = Field(choices=[4,5,6])

    parser = to_argparser(MyArguments)
    args = parser.parse_args([])
    assert args.x == None
    assert args.y == None

    args = parser.parse_args(["--x", "2"])
    assert args.x == 2
    assert args.y == None


def test_action():
    @dataparser
    class MyArguments:
        x: bool = Field(action="store_true")
        y: bool = Field(action="store_false")

    parser = to_argparser(MyArguments)
    args = parser.parse_args([])
    assert not args.x
    assert args.y

    args = parser.parse_args(["--x", "--y"])
    assert args.x
    assert not args.y
