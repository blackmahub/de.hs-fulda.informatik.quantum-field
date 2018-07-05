import ActionCalc as ac
import Plotter as pt
import pandas as pd

import argparse

def main():

    calc = ac.ActionCal()
    plot = pt.PlotGraphs()

    CONST_m = 1

    parser = argparse.ArgumentParser(description="Calculate action of 4D Space Time Histories")
    parser.add_argument('-n', '--noh', help='Number of random space time histories to generate', type=int, default=50)
    parser.add_argument('-dim', '--dimensions', help='Sizes for each of the 4 dimensions', type=int, nargs=4, default=[20,20,20,20], metavar=('dim1', 'dim2', 'dim3', 'dim4'))
    parser.add_argument('-f', '--field', help='Field MIN and MAX', type=float, nargs=2, default=[-1,1], metavar=('min', 'max'))
    parser.add_argument('-p', '-printall', help='Print each action while calculating', action='store_true')
    parser.add_argument('-sc', '-showchart', help='Plot chart in the end', action='store_true')
    args = parser.parse_args()

    noh = args.noh
    dim1 = args.dimensions[0]
    dim2 = args.dimensions[1]
    dim3 = args.dimensions[2]
    dim4 = args.dimensions[3]

    field_min = args.field[0]
    field_max = args.field[1]

    print_all = args.p

    show_chart = args.sc

    interesting = calc.calculate_action(CONST_m, field_min, field_max, [noh, dim1,dim2, dim3, dim4])
    # print(interesting)

    # if show_chart:
    plot.plot_graph(interesting)
        


if __name__ == "__main__":
    main()