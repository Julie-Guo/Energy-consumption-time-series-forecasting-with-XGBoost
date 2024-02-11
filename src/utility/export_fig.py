import os


def export_fig(filename):
    date = datetime.date.today().strftime("%d-%m-%Y")
    path = f"../..reports/figures/{date}/"

    if os.path.exists(path):
        plt.savefig(path + filename + ".png", bbox_inches="tight")

    if not os.path.exists(path):
        os.makedirs(path)
        plt.savefig(path + filename + ".png", bbox_inches="tight")

    print(f"Successfully export {filename}")
