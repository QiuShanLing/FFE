from ffe import parse, parse_ffe_array, parse_ffe_dataset, parse_ffe_grid


def test_fast_parser_array_and_dataset(tmp_path):
    ffe_file = tmp_path / "sample.ffe"
    ffe_file.write_text(
        """#Configuration Name: first
#Frequency: 100
#No. of Theta Samples: 2
#No. of Phi Samples: 2
# "Theta" "Phi" "Re(Etheta)" "Im(Etheta)" "Re(Ephi)" "Im(Ephi)"
 0 0 1 0 2 0
 0 90 3 0 4 0
 90 0 5 0 6 0
 90 90 7 0 8 0
#Configuration Name: second
#Frequency: 200
#No. of Theta Samples: 2
#No. of Phi Samples: 2
# "Theta" "Phi" "Re(Etheta)" "Im(Etheta)" "Re(Ephi)" "Im(Ephi)"
 0 0 11 0 12 0
 0 90 13 0 14 0
 90 0 15 0 16 0
 90 90 17 0 18 0
""",
        encoding="utf-8",
    )

    headers, frequencies, data = parse_ffe_array(ffe_file)
    assert headers == ("Theta", "Phi", "Re(Etheta)", "Im(Etheta)", "Re(Ephi)", "Im(Ephi)")
    assert frequencies.tolist() == [100.0, 200.0]
    assert data.shape == (2, 4, 6)

    grid_headers, grid_frequencies, theta, phi, grid_data = parse_ffe_grid(ffe_file)
    assert grid_headers == headers
    assert grid_frequencies.tolist() == [100.0, 200.0]
    assert theta.tolist() == [0.0, 90.0]
    assert phi.tolist() == [0.0, 90.0]
    assert grid_data.shape == (2, 2, 2, 6)
    assert grid_data[0, 1, 1, 2] == 7.0

    ds = parse_ffe_dataset(ffe_file)
    assert ds["Re(Etheta)"].sel(Frequency=100, Theta=90, Phi=90).item() == 7.0

    ffd = parse(ffe_file)
    assert ffd.electric_field["Etheta"].sel(Frequency=200, Theta=0, Phi=90).item() == 13 + 0j
