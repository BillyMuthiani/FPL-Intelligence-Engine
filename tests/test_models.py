"""Tests for data models."""

import pytest

from fpl_engine.models import (
    Fixture,
    FixtureDifficulty,
    Gameweek,
    Player,
    PlayerPosition,
    PlayerStatus,
    Squad,
    SquadPlayer,
    Team,
)


class TestPlayerPosition:
    """Tests for PlayerPosition enum."""

    def test_from_id(self) -> None:
        assert PlayerPosition.from_id(1) == PlayerPosition.GOALKEEPER
        assert PlayerPosition.from_id(2) == PlayerPosition.DEFENDER
        assert PlayerPosition.from_id(3) == PlayerPosition.MIDFIELDER
        assert PlayerPosition.from_id(4) == PlayerPosition.FORWARD

    def test_from_id_invalid(self) -> None:
        with pytest.raises(ValueError):
            PlayerPosition.from_id(5)

    def test_position_id_property(self) -> None:
        assert PlayerPosition.GOALKEEPER.position_id == 1
        assert PlayerPosition.DEFENDER.position_id == 2
        assert PlayerPosition.MIDFIELDER.position_id == 3
        assert PlayerPosition.FORWARD.position_id == 4


class TestPlayer:
    """Tests for Player model."""

    def test_create_minimal_player(self) -> None:
        """Test creating a player with minimal required fields."""
        player = Player(
            id=1,
            web_name="Test",
            first_name="Test",
            second_name="Player",
            team_id=1,
            position=PlayerPosition.MIDFIELDER,
            status=PlayerStatus.AVAILABLE,
            now_cost=50,
            code=12345,
            element_type=3,
            team_code=1,
        )
        assert player.id == 1
        assert player.full_name == "Test Player"
        assert player.price == 5.0

    def test_position_validation_from_int(self) -> None:
        """Test position validation from integer."""
        player = Player(
            id=1,
            web_name="Test",
            first_name="Test",
            second_name="Player",
            team_id=1,
            position=3,  # int
            status=PlayerStatus.AVAILABLE,
            now_cost=50,
            code=12345,
            element_type=3,
            team_code=1,
        )
        assert player.position == PlayerPosition.MIDFIELDER

    def test_position_validation_from_string(self) -> None:
        """Test position validation from string."""
        player = Player(
            id=1,
            web_name="Test",
            first_name="Test",
            second_name="Player",
            team_id=1,
            position="MID",  # string
            status=PlayerStatus.AVAILABLE,
            now_cost=50,
            code=12345,
            element_type=3,
            team_code=1,
        )
        assert player.position == PlayerPosition.MIDFIELDER

    def test_invalid_position(self) -> None:
        """Test invalid position raises error."""
        with pytest.raises(ValueError):
            Player(
                id=1,
                web_name="Test",
                first_name="Test",
                second_name="Player",
                team_id=1,
                position=99,
                status=PlayerStatus.AVAILABLE,
                now_cost=50,
                code=12345,
                element_type=3,
                team_code=1,
            )

    def test_is_available(self) -> None:
        """Test is_available property."""
        available = Player(
            id=1,
            web_name="Test",
            first_name="Test",
            second_name="Player",
            team_id=1,
            position=PlayerPosition.MIDFIELDER,
            status=PlayerStatus.AVAILABLE,
            now_cost=50,
            code=12345,
            element_type=3,
            team_code=1,
        )
        injured = Player(
            id=2,
            web_name="Test2",
            first_name="Test",
            second_name="Player2",
            team_id=1,
            position=PlayerPosition.MIDFIELDER,
            status=PlayerStatus.INJURED,
            now_cost=50,
            code=12346,
            element_type=3,
            team_code=1,
        )
        assert available.is_available is True
        assert injured.is_available is False

    def test_position_properties(self) -> None:
        """Test position boolean properties."""
        gk = Player(id=1, web_name="GK", first_name="G", second_name="K", team_id=1, position=PlayerPosition.GOALKEEPER, status=PlayerStatus.AVAILABLE, now_cost=40, code=1, element_type=1, team_code=1)
        assert gk.is_goalkeeper is True
        assert gk.is_defender is False

        df = Player(id=2, web_name="DF", first_name="D", second_name="F", team_id=1, position=PlayerPosition.DEFENDER, status=PlayerStatus.AVAILABLE, now_cost=50, code=2, element_type=2, team_code=1)
        assert df.is_defender is True

        md = Player(id=3, web_name="MD", first_name="M", second_name="D", team_id=1, position=PlayerPosition.MIDFIELDER, status=PlayerStatus.AVAILABLE, now_cost=60, code=3, element_type=3, team_code=1)
        assert md.is_midfielder is True

        fw = Player(id=4, web_name="FW", first_name="F", second_name="W", team_id=1, position=PlayerPosition.FORWARD, status=PlayerStatus.AVAILABLE, now_cost=70, code=4, element_type=4, team_code=1)
        assert fw.is_forward is True


class TestTeam:
    """Tests for Team model."""

    def test_create_team(self) -> None:
        """Test creating a team."""
        team = Team(
            id=1,
            code=3,
            name="Arsenal",
            short_name="ARS",
            strength=4,
        )
        assert team.id == 1
        assert team.name == "Arsenal"

    def test_team_equality(self) -> None:
        """Test team equality based on ID."""
        team1 = Team(id=1, code=3, name="Arsenal", short_name="ARS", strength=4)
        team2 = Team(id=1, code=3, name="Arsenal", short_name="ARS", strength=4)
        team3 = Team(id=2, code=4, name="Chelsea", short_name="CHE", strength=4)

        assert team1 == team2
        assert team1 != team3

    def test_team_hash(self) -> None:
        """Test team hashing."""
        team = Team(id=1, code=3, name="Arsenal", short_name="ARS", strength=4)
        team_set = {team}
        assert team in team_set


class TestFixture:
    """Tests for Fixture model."""

    def test_create_fixture(self) -> None:
        """Test creating a fixture."""
        fixture = Fixture(
            id=1,
            code=12345,
            event=1,
            team_h=1,
            team_a=2,
            kickoff_time="2024-08-17T14:00:00Z",
        )
        assert fixture.id == 1
        assert fixture.event == 1

    def test_get_difficulty(self) -> None:
        """Test getting difficulty for a team."""
        fixture = Fixture(
            id=1,
            code=12345,
            event=1,
            team_h=1,
            team_a=2,
            team_h_difficulty=2,
            team_a_difficulty=4,
        )
        assert fixture.get_difficulty(1, True) == FixtureDifficulty.EASY
        assert fixture.get_difficulty(2, False) == FixtureDifficulty.HARD

    def test_get_opponent(self) -> None:
        """Test getting opponent team ID."""
        fixture = Fixture(
            id=1,
            code=12345,
            event=1,
            team_h=1,
            team_a=2,
        )
        assert fixture.get_opponent_id(1) == 2
        assert fixture.get_opponent_id(2) == 1
        assert fixture.get_opponent_id(3) is None

    def test_is_team_home_away(self) -> None:
        """Test home/away checks."""
        fixture = Fixture(id=1, code=12345, event=1, team_h=1, team_a=2)
        assert fixture.is_team_home(1) is True
        assert fixture.is_team_home(2) is False
        assert fixture.is_team_away(2) is True


class TestFixtureDifficulty:
    """Tests for FixtureDifficulty enum."""

    def test_from_fdr(self) -> None:
        assert FixtureDifficulty.from_fdr(1) == FixtureDifficulty.VERY_EASY
        assert FixtureDifficulty.from_fdr(2) == FixtureDifficulty.EASY
        assert FixtureDifficulty.from_fdr(3) == FixtureDifficulty.MEDIUM
        assert FixtureDifficulty.from_fdr(4) == FixtureDifficulty.HARD
        assert FixtureDifficulty.from_fdr(5) == FixtureDifficulty.VERY_HARD

    def test_default_medium(self) -> None:
        assert FixtureDifficulty.from_fdr(99) == FixtureDifficulty.MEDIUM

    def test_numeric_property(self) -> None:
        assert FixtureDifficulty.VERY_EASY.numeric == 1
        assert FixtureDifficulty.HARD.numeric == 4


class TestGameweek:
    """Tests for Gameweek model."""

    def test_create_gameweek(self) -> None:
        """Test creating a gameweek."""
        gw = Gameweek(
            id=1,
            name="Gameweek 1",
            deadline_time="2024-08-16T17:30:00Z",
            deadline_time_epoch=1723829400,
            deadline_time_game_offset=0,
            is_current=True,
        )
        assert gw.id == 1
        assert gw.is_active is True

    def test_gameweek_comparison(self) -> None:
        """Test gameweek ordering."""
        gw1 = Gameweek(id=1, name="GW1", deadline_time="2024-08-16T17:30:00Z", deadline_time_epoch=1, deadline_time_game_offset=0)
        gw2 = Gameweek(id=2, name="GW2", deadline_time="2024-08-23T17:30:00Z", deadline_time_epoch=2, deadline_time_game_offset=0)
        assert gw1 < gw2


class TestSquadPlayer:
    """Tests for SquadPlayer model."""

    def test_create_squad_player(self, sample_player: Player) -> None:
        """Test creating a squad player."""
        sp = SquadPlayer(
            player=sample_player,
            position=1,
            is_captain=True,
            multiplier=1,
        )
        assert sp.position == 1
        assert sp.is_captain is True
        assert sp.multiplier == 1

    def test_invalid_position(self, sample_player: Player) -> None:
        """Test invalid squad position."""
        with pytest.raises(ValueError):
            SquadPlayer(player=sample_player, position=0)

        with pytest.raises(ValueError):
            SquadPlayer(player=sample_player, position=16)

    def test_invalid_multiplier(self, sample_player: Player) -> None:
        """Test invalid multiplier."""
        with pytest.raises(ValueError):
            SquadPlayer(player=sample_player, position=1, multiplier=4)


class TestSquad:
    """Tests for Squad model."""

    def test_create_squad(self, sample_player: Player) -> None:
        """Test creating a squad."""
        players = [
            SquadPlayer(player=sample_player, position=i)
            for i in range(1, 16)
        ]
        squad = Squad(players=players, gameweek=1, bank=5.0)
        assert squad.starting_xi_count == 11
        assert squad.bench_count == 4

    def test_squad_size_validation(self, sample_player: Player) -> None:
        """Test squad size validation."""
        # Creating 16 SquadPlayers will fail at SquadPlayer level (position 16 invalid)
        with pytest.raises(Exception):  # Pydantic raises ValidationError
            [SquadPlayer(player=sample_player, position=i) for i in range(1, 17)]

    def test_get_starting_xi(self, sample_player: Player) -> None:
        """Test getting starting XI."""
        players = [SquadPlayer(player=sample_player, position=i) for i in range(1, 16)]
        squad = Squad(players=players, gameweek=1)
        xi = squad.get_starting_xi()
        assert len(xi) == 11

    def test_get_bench(self, sample_player: Player) -> None:
        """Test getting bench."""
        players = [SquadPlayer(player=sample_player, position=i) for i in range(1, 16)]
        squad = Squad(players=players, gameweek=1)
        bench = squad.get_bench()
        assert len(bench) == 4

    def test_get_captain(self, sample_player: Player) -> None:
        """Test getting captain."""
        players = [
            SquadPlayer(player=sample_player, position=1, is_captain=True),
            SquadPlayer(player=sample_player, position=2),
        ]
        squad = Squad(players=players, gameweek=1)
        captain = squad.get_captain()
        assert captain is not None
        assert captain.is_captain is True

    def test_validate_formation_valid(self) -> None:
        """Test valid formation validation."""
        # Create players for each position, spread across clubs to respect 3-player limit
        # Starting XI: 1 GK, 3 DEF, 4 MID, 3 FWD = 11
        # Bench: 1 GK, 2 DEF, 1 MID, 1 FWD = 4
        gk1 = Player(id=1, web_name="GK1", first_name="GK", second_name="1", team_id=1, position=PlayerPosition.GOALKEEPER, status=PlayerStatus.AVAILABLE, now_cost=40, code=1, element_type=1, team_code=1)
        gk2 = Player(id=2, web_name="GK2", first_name="GK", second_name="2", team_id=2, position=PlayerPosition.GOALKEEPER, status=PlayerStatus.AVAILABLE, now_cost=40, code=2, element_type=1, team_code=2)

        defs = [Player(id=i, web_name=f"DF{i}", first_name="DF", second_name=str(i), team_id=i, position=PlayerPosition.DEFENDER, status=PlayerStatus.AVAILABLE, now_cost=50, code=i+10, element_type=2, team_code=i) for i in range(3, 8)]
        mids = [Player(id=i, web_name=f"MD{i}", first_name="MD", second_name=str(i), team_id=i, position=PlayerPosition.MIDFIELDER, status=PlayerStatus.AVAILABLE, now_cost=60, code=i+20, element_type=3, team_code=i) for i in range(8, 13)]
        fwds = [Player(id=i, web_name=f"FW{i}", first_name="FW", second_name=str(i), team_id=i, position=PlayerPosition.FORWARD, status=PlayerStatus.AVAILABLE, now_cost=70, code=i+30, element_type=4, team_code=i) for i in range(13, 16)]

        all_players = [gk1, gk2] + defs + mids + fwds

        # Assign positions carefully: 1 GK, 3 DEF, 4 MID, 3 FWD in starting XI
        # Position 1: GK1
        # Positions 2-4: DF3, DF4, DF5
        # Positions 5-8: MD8, MD9, MD10, MD11
        # Positions 9-11: FW13, FW14, FW15
        # Bench (12-15): GK2, DF6, DF7, MD12
        squad_players = [
            SquadPlayer(player=gk1, position=1),  # GK
            SquadPlayer(player=defs[0], position=2),  # DF
            SquadPlayer(player=defs[1], position=3),  # DF
            SquadPlayer(player=defs[2], position=4),  # DF
            SquadPlayer(player=mids[0], position=5),  # MID
            SquadPlayer(player=mids[1], position=6),  # MID
            SquadPlayer(player=mids[2], position=7),  # MID
            SquadPlayer(player=mids[3], position=8),  # MID
            SquadPlayer(player=fwds[0], position=9),  # FWD
            SquadPlayer(player=fwds[1], position=10),  # FWD
            SquadPlayer(player=fwds[2], position=11),  # FWD
            SquadPlayer(player=gk2, position=12),  # Bench GK
            SquadPlayer(player=defs[3], position=13),  # Bench DEF
            SquadPlayer(player=defs[4], position=14),  # Bench DEF
            SquadPlayer(player=mids[4], position=15),  # Bench MID
        ]
        squad = Squad(players=squad_players, gameweek=1)

        valid, errors = squad.validate_formation()
        assert valid is True
        assert len(errors) == 0

        # Also check club constraints
        valid2, errors2 = squad.validate_club_constraints()
        assert valid2 is True
        assert len(errors2) == 0

    def test_validate_formation_invalid(self, sample_player: Player) -> None:
        """Test invalid formation validation."""
        # All midfielders - invalid
        players = [SquadPlayer(player=sample_player, position=i) for i in range(1, 16)]
        squad = Squad(players=players, gameweek=1)
        valid, errors = squad.validate_formation()
        assert valid is False
        assert len(errors) > 0

    def test_validate_club_constraints(self) -> None:
        """Test club constraint validation."""
        # 4 players from same club
        players = []
        for i in range(4):
            p = Player(
                id=i+1,
                web_name=f"P{i}",
                first_name="P",
                second_name=str(i),
                team_id=1,  # Same club
                position=PlayerPosition.MIDFIELDER,
                status=PlayerStatus.AVAILABLE,
                now_cost=50,
                code=100+i,
                element_type=3,
                team_code=1,
            )
            players.append(SquadPlayer(player=p, position=i+1))
        # Add rest from other clubs
        for i in range(4, 15):
            p = Player(
                id=i+1,
                web_name=f"P{i}",
                first_name="P",
                second_name=str(i),
                team_id=i+1,
                position=PlayerPosition.MIDFIELDER,
                status=PlayerStatus.AVAILABLE,
                now_cost=50,
                code=200+i,
                element_type=3,
                team_code=i+1,
            )
            players.append(SquadPlayer(player=p, position=i+1))

        squad = Squad(players=players, gameweek=1)
        valid, errors = squad.validate_club_constraints()
        assert valid is False
        assert any("max 3" in e for e in errors)

    def test_formation_string(self) -> None:
        """Test formation string generation."""
        gk = Player(id=1, web_name="GK", first_name="G", second_name="K", team_id=1, position=PlayerPosition.GOALKEEPER, status=PlayerStatus.AVAILABLE, now_cost=40, code=1, element_type=1, team_code=1)
        defs = [Player(id=i, web_name=f"DF{i}", first_name="DF", second_name=str(i), team_id=1, position=PlayerPosition.DEFENDER, status=PlayerStatus.AVAILABLE, now_cost=50, code=i, element_type=2, team_code=1) for i in range(2, 5)]
        mids = [Player(id=i, web_name=f"MD{i}", first_name="MD", second_name=str(i), team_id=2, position=PlayerPosition.MIDFIELDER, status=PlayerStatus.AVAILABLE, now_cost=60, code=i, element_type=3, team_code=2) for i in range(5, 9)]
        fwds = [Player(id=i, web_name=f"FW{i}", first_name="FW", second_name=str(i), team_id=3, position=PlayerPosition.FORWARD, status=PlayerStatus.AVAILABLE, now_cost=70, code=i, element_type=4, team_code=3) for i in range(9, 12)]
        bench = [
            Player(id=12, web_name="GK2", first_name="GK", second_name="2", team_id=4, position=PlayerPosition.GOALKEEPER, status=PlayerStatus.AVAILABLE, now_cost=40, code=12, element_type=1, team_code=4),
            Player(id=13, web_name="DF5", first_name="DF", second_name="5", team_id=5, position=PlayerPosition.DEFENDER, status=PlayerStatus.AVAILABLE, now_cost=50, code=13, element_type=2, team_code=5),
            Player(id=14, web_name="MD5", first_name="MD", second_name="5", team_id=6, position=PlayerPosition.MIDFIELDER, status=PlayerStatus.AVAILABLE, now_cost=60, code=14, element_type=3, team_code=6),
            Player(id=15, web_name="FW3", first_name="FW", second_name="3", team_id=7, position=PlayerPosition.FORWARD, status=PlayerStatus.AVAILABLE, now_cost=70, code=15, element_type=4, team_code=7),
        ]

        all_players = [gk] + defs + mids + fwds + bench
        squad_players = [SquadPlayer(player=p, position=i+1) for i, p in enumerate(all_players)]
        squad = Squad(players=squad_players, gameweek=1)
        assert squad.get_formation_string() == "3-4-3"
