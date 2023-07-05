from gpt_zola_book_kt import Gen, rprint, torch

if __name__ == "__main__":
    gen = Gen("state_zb_20230406.tar")
    args_l = [
        [
            "D'ailleurs, comme par un contrecoup fatal, tout se gâtait de même\nautour des "
            "Roubaud. Une nouvelle bourrasque de commérages et de\ndiscussions soufflait dans "
            "le couloir.  Philomène venait de\nrompre violemment avec madame Lebleu, à la suite"
            " d'une calomnie\nde cette",
            "zola/v1/pg5250.txt",
            8,
        ],
        [
            "D'ailleurs, comme par un contrecoup fatal, tout se gâtait de même\nautour des "
            "Roubaud. Une nouvelle bourrasque de commérages et de\ndiscussions soufflait dans "
            "le couloir.  Philomène venait de\nrompre violemment avec madame Lebleu, à la suite"
            " d'une calomnie\nde cette",
            "zola/v1/pg5154.txt", 8
        ],
        [
            "D'ailleurs, comme par un contrecoup fatal, tout se gâtait de même\nautour des "
            "Roubaud. Une nouvelle bourrasque de commérages et de\ndiscussions soufflait dans "
            "le couloir.  Philomène venait de\nrompre violemment avec madame Lebleu, à la suite"
            " d'une calomnie\nde cette ",
            "zola/v1/pg5250.txt", 8
        ],
        [
            "D'ailleurs, comme par un contrecoup fatal, tout se gâtait de même\nautour des "
            "Roubaud. Une nouvelle bourrasque de commérages et de\ndiscussions soufflait dans "
            "le couloir.  Philomène venait de\nrompre violemment avec madame Lebleu, à la suite"
            " d'une calomnie\nde cette ",
            "zola/v1/pg5154.txt", 8
        ],
        [
            "D'ailleurs, comme par un contrecoup fatal, tout se gâtait de même\nautour des "
            "Roubaud. Une nouvelle bourrasque de commérages et de\ndiscussions soufflait dans "
            "le couloir.  Philomène venait de\nrompre violemment avec madame Lebleu, à la suite"
            " d'une calomnie\nde cette d",
            "zola/v1/pg5250.txt", 8
        ],
        [
            "D'ailleurs, comme par un contrecoup fatal, tout se gâtait de même\nautour des "
            "Roubaud. Une nouvelle bourrasque de commérages et de\ndiscussions soufflait dans "
            "le couloir.  Philomène venait de\nrompre violemment avec madame Lebleu, à la suite"
            " d'une calomnie\nde cette d",
            "zola/v1/pg5154.txt", 8
        ],
        [
            "D'ailleurs, comme par un contrecoup fatal, tout se gâtait de même\nautour des "
            "Roubaud. Une nouvelle bourrasque de commérages et de\ndiscussions soufflait dans "
            "le couloir.  Philomène venait de\nrompre violemment avec madame Lebleu, à la suite"
            " d'une calomnie\nde cette de",
            "zola/v1/pg5250.txt", 8
        ],
        [
            "D'ailleurs, comme par un contrecoup fatal, tout se gâtait de même\nautour des "
            "Roubaud. Une nouvelle bourrasque de commérages et de\ndiscussions soufflait dans "
            "le couloir.  Philomène venait de\nrompre violemment avec madame Lebleu, à la suite"
            " d'une calomnie\nde cette de",
            "zola/v1/pg5154.txt", 8
        ],
        [
            "D'ailleurs, comme par un contrecoup fatal, tout se gâtait de même\nautour des "
            "Roubaud. Une nouvelle bourrasque de commérages et de\ndiscussions soufflait dans "
            "le couloir.  Philomène venait de\nrompre violemment avec madame Lebleu, à la suite"
            " d'une calomnie\nde cette dernière",
            "zola/v1/pg5154.txt", 8
        ],
    ]
    for args in args_l:
        ret = gen.get_next_word_list(*args)
        # print(ret)
