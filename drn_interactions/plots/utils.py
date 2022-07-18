import textwrap
import matplotlib as mpl


def wrap_xlabels(ax, width, break_long_words=False):
    labels = []
    for label in ax.get_xticklabels():
        text = label.get_text()
        labels.append(
            textwrap.fill(text, width=width, break_long_words=break_long_words)
        )
    ax.set_xticklabels(labels, rotation=0)


def wrap_ylabels(ax, width, break_long_words=False):
    labels = []
    for label in ax.get_yticklabels():
        text = label.get_text()
        labels.append(
            textwrap.fill(text, width=width, break_long_words=break_long_words)
        )
    ax.set_yticklabels(labels, rotation=0)


def wrap_xlabel(ax, width, break_long_words=False):
    label = ax.get_xlabel()
    ax.set_xlabel(textwrap.fill(label, width=width, break_long_words=break_long_words))


def wrap_ylabel(ax, width, break_long_words=False):
    label = ax.get_ylabel()
    ax.set_ylabel(textwrap.fill(label, width=width, break_long_words=break_long_words))


def wrap_title(ax, width, break_long_words=False):
    label = ax.get_title()
    ax.set_title(textwrap.fill(label, width=width, break_long_words=break_long_words))


def wrap_all_text(ax, width, break_long_words=False):
    wrap_xlabels(ax, width, break_long_words)
    wrap_ylabels(ax, width, break_long_words)
    wrap_xlabel(ax, width, break_long_words)
    wrap_ylabel(ax, width, break_long_words)
    wrap_title(ax, width, break_long_words)
    return ax
