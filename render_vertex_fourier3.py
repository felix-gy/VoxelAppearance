from src.vertex_appearance.runtime import parser, render

if __name__ == '__main__':
    render(parser().parse_args())
