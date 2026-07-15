from electricity_demand.data import download_power_data


def main():
    """
    Download the raw OPSD electricity-demand dataset.
    """

    output_path = download_power_data()

    print(
        f"Electricity data are available at: "
        f"{output_path}"
    )


if __name__ == "__main__":
    main()