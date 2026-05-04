{ pkgs }:
let
  python = pkgs.python3.withPackages (ps: with ps; [
    matplotlib
    pandas
  ]);

  plotPy = pkgs.writeText "plot.py" ''
    import argparse
    import datetime
    import sys
    from pathlib import Path
    import pandas as pd
    import matplotlib.pyplot as plt

    """
    Example:
      plot --csv tessa.csv storm-add.csv storm-spm.csv \
           --label Tessa "Storm (ADD)" "Storm (SPM)" \
           --x N --y run_mean time time \
           --yscale log --title "Herman — scaling with N"
    """


    def column(df, key, idx):
        return df[key] if key else df.iloc[:, idx]


    def broadcast(lst, n):
        """Extend lst to length n by repeating the last element."""
        if not lst:
            return [None] * n
        return lst + [lst[-1]] * (n - len(lst))


    if __name__ == "__main__":

        parser = argparse.ArgumentParser()
        parser.add_argument("--csv", type=str, nargs="+", required=True)
        parser.add_argument("--label", type=str, nargs="+", required=True)
        parser.add_argument("--color", type=str, nargs="*")
        parser.add_argument("--title", type=str)
        parser.add_argument("--savefig", type=str)
        parser.add_argument("--ylim", type=float, nargs="*")
        parser.add_argument("--xlim", type=float, nargs="*")
        parser.add_argument(
            "--x", type=str, nargs="*",
            help="X column(s). One per CSV, or broadcast.",
        )
        parser.add_argument(
            "--y", type=str, nargs="*",
            help="Y column(s). One per CSV, or broadcast.",
        )
        parser.add_argument("--xlabel", type=str)
        parser.add_argument("--ylabel", type=str)
        parser.add_argument(
            "--yscale",
            type=str,
            choices=["linear", "log", "symlog", "logit"],
            default="linear",
        )
        parser.add_argument(
            "--xscale",
            type=str,
            choices=["linear", "log", "symlog", "logit"],
            default="linear",
        )
        parser.add_argument("--marker", type=str, default=None,
                            help="Marker style, e.g. 'o'")
        parser.add_argument("--markersize", type=float, default=4)
        parser.add_argument("--grid", action="store_true", default=False)
        parser.add_argument("--figsize", type=float, nargs=2, default=[8, 5])

        args = parser.parse_args()
        n = len(args.csv)
        assert len(args.label) == n, f"need {n} labels, got {len(args.label)}"
        assert args.color is None or len(args.color) == n
        assert args.ylim is None or len(args.ylim) == 2
        assert args.xlim is None or len(args.xlim) == 2

        x_cols = broadcast(args.x, n)
        y_cols = broadcast(args.y, n)
        colors = [None] * n if args.color is None else args.color

        fig, ax = plt.subplots(figsize=tuple(args.figsize))

        for csv_path, label, x_col, y_col, color in zip(
            args.csv, args.label, x_cols, y_cols, colors
        ):
            try:
                df = pd.read_csv(csv_path)
            except (FileNotFoundError, pd.errors.EmptyDataError) as exc:
                print(f"[plot] WARNING: skipping {csv_path}: {exc}", file=sys.stderr)
                continue
            x = column(df, x_col, 0)
            y = column(df, y_col, 1)
            kwargs = {"label": label}
            if color:
                kwargs["color"] = color
            if args.marker:
                kwargs["marker"] = args.marker
                kwargs["markersize"] = args.markersize
            ax.plot(x, y, **kwargs)

        if args.title:
            ax.set_title(args.title)
        ax.set_xlabel(args.xlabel or (x_cols[0] if x_cols[0] else ""))
        ax.set_ylabel(args.ylabel or (y_cols[0] if y_cols[0] else "Time (s)"))
        ax.legend(fontsize="small")

        if args.ylim is not None:
            ax.set_ylim(args.ylim[0], args.ylim[1])
        if args.xlim is not None:
            ax.set_xlim(args.xlim[0], args.xlim[1])
        ax.set_yscale(args.yscale)
        ax.set_xscale(args.xscale)

        if args.grid:
            ax.grid(True, alpha=0.3)

        fig.tight_layout()

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = Path(args.csv[0]).stem if n == 1 else "plot"
        output_filename = args.savefig if args.savefig \
            else f"{filename}_{timestamp}.png"
        fig.savefig(output_filename, dpi=150)
        print(f"  plot {output_filename}")
  '';

in
# python3 -I (isolated mode) ignores PYTHON* env vars (so the dev shell's
# PYTHONPATH=$PWD does not pull /home/.../sitecustomize.py in front of the
# nix env's, which would otherwise prevent NIX_PYTHONPATH from injecting
# matplotlib/pandas onto sys.path). NIX_PYTHONPATH itself is preserved by -I
# because it is not a PYTHON* variable.
pkgs.writeShellScriptBin "plot" ''
  exec ${python}/bin/python3 -I ${plotPy} "$@"
''
