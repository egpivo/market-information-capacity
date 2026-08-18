//! The shipped TOML files and the in-code configurations must not drift apart.

use market_information_capacity::config::Config;

#[test]
fn publication_toml_matches_the_canonical_configuration() {
    let from_file = Config::load("configs/publication.toml").expect("publication.toml loads");
    assert_eq!(
        from_file,
        Config::publication(),
        "configs/publication.toml has drifted from Config::publication()"
    );
}

#[test]
fn quick_toml_matches_the_reduced_scale_configuration() {
    let from_file = Config::load("configs/quick.toml").expect("quick.toml loads");
    assert_eq!(
        from_file,
        Config::quick(),
        "configs/quick.toml has drifted from Config::quick()"
    );
}

#[test]
fn quick_and_publication_differ_only_in_monte_carlo_scale() {
    let quick = Config::quick();
    let publication = Config::publication();
    assert_eq!(quick.master_seed, publication.master_seed);
    assert_eq!(quick.worlds.presets, publication.worlds.presets);
    assert_eq!(quick.grid.k_values, publication.grid.k_values);
    assert_eq!(quick.grid.trader_values, publication.grid.trader_values);
    assert_eq!(quick.asymptote.k_values, publication.asymptote.k_values);
    assert_eq!(quick.same_price.band, publication.same_price.band);
    assert!(quick.asymptote.realizations < publication.asymptote.realizations);
}
