import numpy as np

# [dep/fac][U][tau] [Vdtc/Vstdf/Vfac]
STPCalibData = {
    "dep":
        {
            1: np.array([
                []
            ]).transpose(),

            3: np.array([
                # tau, Vdtc, Vstdf, Vfac
                [90., 2.5, 1.3, 0.1],
                [130., 1.9, 1.3, 0.1],
                [180., 0.7, 0.7, 0.1],
                [300., 0.7, 1.3, 0.1],
            ]).transpose(),

            5: np.array([
                # tau, Vdtc, Vstdf, Vfac
                [80., 2.5, 0.7, 0.1],
                [100., 0.7, 0.45, 0.1],  # nice
                [140., 1.3, 0.8, 0.1],  # nice
                [180., 0.7, 0.7, 0.1],
                [200., 1.3, 1.3, 0.1],
                [220., 1.3, 1.15, 0.1],
                [250., 1.3, 1.5, 0.1],
                [360., 0.7, 1.15, 0.1],
                [450., 0.7, 1.5, 0.1],
            ]).transpose(),

            7: np.array([
                # tau, Vdtc, Vstdf, Vfac
                [40., 2.1, 0.7, 0.1],
                [50., 1.8, 0.7, 0.1],
                [80., 2.5, 0.8, 0.1],
                [110., 1.9, 0.8, 0.1],  # nice
                [150., 1.3, 0.8, 0.1],
                [220., 1.3, 1.15, 0.1],
                [250., 0.7, 0.8, 0.1],
                [380., 0.7, 1.15, 0.1],
                [900., 0.1, 0.45, 0.1],
            ]).transpose(),
        },
    "fac":
        {
            1: np.array([
                # tau, Vdtc, Vstdf, Vfac
                [70., 0.1, 0.8, 1.5],
                [100., 0.1, 0.1, 1.5],
                [130., 2.0, 1.5, 1.5],  # nice
                [150., 0.1, 0.45, 1.5],
                [190., 0.73, 1.15, 2.0],
            ]).transpose(),

            3: np.array([
                # tau, Vdtc, Vstdf, Vfac
                [35., 0.1, 0.5, 2.4],
                [50., 0.6, 0.5, 2.4],
                [90., 0.73, 0.1, 2.0],
                [100., 0.1, 0.1, 2.0],
                [130., 0.1, 0.45, 2.0],
                [150., 1.37, 1.5, 1.5],
            ]).transpose(),

            5: np.array([
                # tau, Vdtc, Vstdf, Vfac
                [90., 1.37, 0.8, 2.5],  # nice
                [110., 2.0, 1.5, 2.0],
                [150., 2.0, 0.8, 1.5],
                [200., 0.73, 1.15, 2.0],
            ]).transpose(),

            7: np.array([
                # tau, Vdtc, Vstdf, Vfac
                [90., 1.37, 0.45, 1.5],  # nice
                [100., 2.0, 1.5, 2.0],  # nice
                [150., 1.37, 1.5, 2.5],
                [180., 0.73, 1.15, 2.0],
            ]).transpose(),
        }
}


def clip(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx

""" returns nearest: tau, Vdtc, Vstdf, Vfac """


def get_stp_params(stp_type, U, tau):
    data = STPCalibData[stp_type][U]
    return data[:, clip(data[0], tau)]

if __name__ == "__main__":
    print get_stp_params('fac', 1, 120)
