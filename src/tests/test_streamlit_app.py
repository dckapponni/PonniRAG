"""
Enhanced Test Suite for Ponni Archive Streamlit Application
Target: 95%+ test coverage

Adds coverage for previously missing lines:
- Error handling paths
- Edge cases in image loading
- Session state deletion scenarios
- Query parameter edge cases
- PDF viewer error scenarios
- About page image loading variations
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, call
from PIL import Image
import io

sys.path.append(str(Path(__file__).resolve().parents[1]))


class TestConfigurePage:
    """Test suite for page configuration."""
    
    @patch('streamlit_app.st')
    @patch('streamlit_app.logger')
    def test_configure_page_sets_all_params(self, mock_logger, mock_st):
        """Test configure_page sets all required parameters."""
        from streamlit_app import configure_page
        
        configure_page()
        
        mock_st.set_page_config.assert_called_once_with(
            page_title="Ponni Archive",
            page_icon="scroll",
            layout="wide",
            initial_sidebar_state="collapsed",
        )


class TestHandleQueryParameters:
    """Test suite for query parameter handling."""
    
    @patch('streamlit_app.st')
    def test_language_switch_tamil_to_english(self, mock_st):
        """Test language switching from Tamil to English."""
        session_state = {}
        
        class SessionStateMock:
            def __contains__(self, key):
                return key in session_state
            
            def __getitem__(self, key):
                return session_state[key]
            
            def __setitem__(self, key, value):
                session_state[key] = value
            
            def __delitem__(self, key):
                del session_state[key]
            
            def __getattr__(self, key):
                if key in session_state:
                    return session_state[key]
                raise AttributeError(f"'{type(self).__name__}' object has no attribute '{key}'")
            
            def __setattr__(self, key, value):
                session_state[key] = value
            
            def get(self, key, default=None):
                return session_state.get(key, default)
        
        mock_st.session_state = SessionStateMock()
        mock_st.query_params.get.side_effect = lambda key, default=None: {
            "page": "home",
            "lang": "en",
        }.get(key, default)
        mock_st.query_params.__contains__ = Mock(side_effect=lambda k: k == "lang")
        mock_st.query_params.__getitem__ = Mock(side_effect=lambda k: {"lang": "en", "page": "home"}[k])
        mock_st.query_params.to_dict.return_value = {"page": "home", "lang": "en"}
        
        from streamlit_app import handle_query_parameters
        
        handle_query_parameters()
        
        assert session_state.get('language') == "en"
    
    @patch('streamlit_app.st')
    def test_pdf_viewer_parameters(self, mock_st):
        """Test PDF viewer page with volume and issue parameters."""
        mock_st.query_params.get.side_effect = lambda key, default=None: {
            "page": "pdf_viewer",
            "volume": "3",
            "issue": "7",
        }.get(key, default)
        mock_st.query_params.__contains__ = Mock(return_value=False)
        mock_st.query_params.to_dict.return_value = {"page": "pdf_viewer", "volume": "3", "issue": "7"}
        
        from streamlit_app import handle_query_parameters
        
        page, volume, issue = handle_query_parameters()
        
        assert page == "pdf_viewer"
        assert volume == "3"
        assert issue == "7"
    
    @patch('streamlit_app.st')
    def test_handle_query_parameters_no_lang_param(self, mock_st):
        """Test query parameters without lang parameter."""
        session_state = {"language": "ta"}
        
        class SessionStateMock:
            def __contains__(self, key):
                return key in session_state
            def get(self, key, default=None):
                return session_state.get(key, default)
        
        mock_st.session_state = SessionStateMock()
        mock_st.query_params.get.side_effect = lambda key, default=None: {
            "page": "library"
        }.get(key, default)
        mock_st.query_params.__contains__ = Mock(return_value=False)
        mock_st.query_params.to_dict.return_value = {"page": "library"}
        
        from streamlit_app import handle_query_parameters
        
        page, volume, issue = handle_query_parameters()
        
        assert page == "library"
        assert volume is None
        assert issue is None


class TestRenderNavigationBar:
    """Test suite for navigation bar rendering."""
    
    @patch('streamlit_app.st')
    def test_navigation_bar_tamil_mode(self, mock_st):
        """Test navigation bar in Tamil mode."""
        mock_st.session_state.language = "ta"
        mock_st.query_params.get.return_value = "home"
        
        from streamlit_app import render_navigation_bar
        
        render_navigation_bar()
        
        mock_st.markdown.assert_called_once()
        nav_html = mock_st.markdown.call_args[0][0]
        
        assert "பொன்னி களஞ்சியம்" in nav_html
        assert "AI-யிடம் கேளுங்கள்" in nav_html
        assert "நூலகம்" in nav_html
        assert "English" in nav_html
    
    @patch('streamlit_app.st')
    def test_navigation_bar_english_mode(self, mock_st):
        """Test navigation bar in English mode."""
        mock_st.session_state.language = "en"
        mock_st.query_params.get.return_value = "library"
        
        from streamlit_app import render_navigation_bar
        
        render_navigation_bar()
        
        nav_html = mock_st.markdown.call_args[0][0]
        
        assert "Ponni Archive" in nav_html
        assert "Ask AI" in nav_html
        assert "Library" in nav_html
        assert "தமிழ்" in nav_html
    
    @patch('streamlit_app.st')
    def test_navigation_bar_with_page_param(self, mock_st):
        """Test navigation bar with different page parameters."""
        mock_st.session_state.language = "en"
        mock_st.query_params.get.return_value = "about"
        
        from streamlit_app import render_navigation_bar
        
        render_navigation_bar()
        
        nav_html = mock_st.markdown.call_args[0][0]
        assert "page=about" in nav_html


class TestHandleSuggestionClick:
    """Test suite for suggestion button handling."""
    
    @patch('streamlit_app.st')
    @patch('streamlit_app.logger')
    def test_suggestion_click_stores_prompt(self, mock_logger, mock_st):
        """Test clicking suggestion stores prompt in session state."""
        mock_st.session_state = MagicMock()
        
        from streamlit_app import handle_suggestion_click
        
        test_prompt = "Test prompt text"
        handle_suggestion_click(test_prompt)
        
        assert mock_st.session_state.temp_submit == test_prompt


class TestRenderHomePage:
    """Test suite for home page rendering."""
    
    @patch('streamlit_app.st')
    @patch('streamlit_app.handle_user_input')
    def test_render_home_page_no_messages(self, mock_handle_input, mock_st):
        """Test home page with no chat messages."""
        mock_st.session_state.messages = []
        mock_st.session_state.language = "ta"
        
        def columns_side_effect(spec, **kwargs):
            if isinstance(spec, int):
                num_cols = spec
            elif isinstance(spec, list):
                num_cols = len(spec)
            else:
                num_cols = spec
            
            cols = []
            for _ in range(num_cols):
                col = MagicMock()
                col.__enter__ = Mock(return_value=col)
                col.__exit__ = Mock(return_value=None)
                cols.append(col)
            return cols
        
        mock_st.columns.side_effect = columns_side_effect
        mock_st.button.return_value = False
        
        from streamlit_app import render_home_page
        
        render_home_page()
        
        mock_st.markdown.assert_called()
        assert mock_st.button.call_count >= 4
    
    @patch('streamlit_app.st')
    @patch('streamlit_app.handle_user_input')
    @patch('streamlit_app.render_sources')
    def test_render_home_page_with_messages(self, mock_render_sources, mock_handle_input, mock_st):
        """Test home page with existing chat messages."""
        mock_st.session_state.messages = [
            {"role": "user", "content": "Test question"},
            {"role": "assistant", "content": "Test answer", "sources": []}
        ]
        mock_st.session_state.language = "en"
        
        def columns_side_effect(spec, **kwargs):
            if isinstance(spec, int):
                num_cols = spec
            elif isinstance(spec, list):
                num_cols = len(spec)
            else:
                num_cols = spec
            
            cols = []
            for _ in range(num_cols):
                col = MagicMock()
                col.__enter__ = Mock(return_value=col)
                col.__exit__ = Mock(return_value=None)
                cols.append(col)
            return cols
        
        mock_st.columns.side_effect = columns_side_effect
        
        mock_chat_message = MagicMock()
        mock_chat_message.__enter__ = Mock(return_value=mock_chat_message)
        mock_chat_message.__exit__ = Mock(return_value=None)
        mock_st.chat_message.return_value = mock_chat_message
        
        from streamlit_app import render_home_page
        
        render_home_page()
        
        mock_handle_input.assert_called_once()
    
    @patch('streamlit_app.st')
    @patch('streamlit_app.handle_user_input')
    def test_render_home_page_button_clicks_trigger_rerun(self, mock_handle_input, mock_st):
        """Test that clicking suggestion buttons triggers rerun."""
        mock_st.session_state.messages = []
        mock_st.session_state.language = "en"
        
        def columns_side_effect(spec, **kwargs):
            if isinstance(spec, int):
                num_cols = spec
            elif isinstance(spec, list):
                num_cols = len(spec)
            else:
                num_cols = spec
            
            cols = []
            for _ in range(num_cols):
                col = MagicMock()
                col.__enter__ = Mock(return_value=col)
                col.__exit__ = Mock(return_value=None)
                cols.append(col)
            return cols
        
        mock_st.columns.side_effect = columns_side_effect
        
        # Simulate button click
        button_click_count = [0]
        def button_side_effect(*args, **kwargs):
            button_click_count[0] += 1
            return button_click_count[0] == 1  # First button returns True
        
        mock_st.button.side_effect = button_side_effect
        
        # Mock st.rerun() to raise exception
        mock_st.rerun.side_effect = Exception("Rerun triggered")
        
        from streamlit_app import render_home_page
        
        with pytest.raises(Exception, match="Rerun triggered"):
            render_home_page()


class TestHandleUserInput:
    """Test suite for user input handling."""
    
    @patch('streamlit_app.st')
    @patch('streamlit_app.ask_question')
    def test_handle_user_input_with_temp_submit(self, mock_ask_question, mock_st):
        """Test handling temporary submit from suggestion."""
        session_state = {"temp_submit": "Test query", "messages": [], "language": "en"}
        
        class SessionStateMock:
            def __contains__(self, key):
                return key in session_state
            
            def __getitem__(self, key):
                return session_state[key]
            
            def __setitem__(self, key, value):
                session_state[key] = value
            
            def __delitem__(self, key):
                del session_state[key]
            
            def __getattr__(self, key):
                if key in session_state:
                    return session_state[key]
                raise AttributeError(f"'{type(self).__name__}' object has no attribute '{key}'")
            
            def __setattr__(self, key, value):
                session_state[key] = value
            
            def __delattr__(self, key):
                if key in session_state:
                    del session_state[key]
                else:
                    raise AttributeError(f"'{type(self).__name__}' object has no attribute '{key}'")
            
            def get(self, key, default=None):
                return session_state.get(key, default)
        
        mock_st.session_state = SessionStateMock()
        mock_st.chat_input.return_value = None
        
        mock_spinner = MagicMock()
        mock_spinner.__enter__ = Mock(return_value=mock_spinner)
        mock_spinner.__exit__ = Mock(return_value=None)
        mock_st.spinner.return_value = mock_spinner
        
        mock_chat_message = MagicMock()
        mock_chat_message.__enter__ = Mock(return_value=mock_chat_message)
        mock_chat_message.__exit__ = Mock(return_value=None)
        mock_st.chat_message.return_value = mock_chat_message
        
        mock_ask_question.return_value = {
            "answer": "Test answer",
            "sources": []
        }
        
        from streamlit_app import handle_user_input
        
        handle_user_input()
        
        assert len(session_state["messages"]) == 2
        assert session_state["messages"][0]["role"] == "user"
        assert session_state["messages"][0]["content"] == "Test query"
        assert session_state["messages"][1]["role"] == "assistant"
        assert "temp_submit" not in session_state


class TestRenderSources:
    """Test suite for source rendering."""
    
    @patch('streamlit_app.st')
    def test_render_sources_with_payload(self, mock_st):
        """Test rendering sources with payload attribute."""
        mock_expander = MagicMock()
        mock_st.expander.return_value.__enter__ = Mock(return_value=mock_expander)
        mock_st.expander.return_value.__exit__ = Mock(return_value=None)
        mock_st.session_state.language = "en"
        
        mock_source = Mock()
        mock_source.payload = {
            "content": "Short content",
            "metadata": {
                "doc_issue": "vol_1_issue_1",
                "doc_id": "1",
                "title": "Test",
                "author_name": "Author"
            }
        }
        
        from streamlit_app import render_sources
        
        render_sources(0, [mock_source])
        
        mock_st.expander.assert_called_once()
    
    @patch('streamlit_app.st')
    def test_render_sources_long_content(self, mock_st):
        """Test rendering sources with long content requiring truncation."""
        mock_expander = MagicMock()
        mock_st.expander.return_value.__enter__ = Mock(return_value=mock_expander)
        mock_st.expander.return_value.__exit__ = Mock(return_value=None)
        mock_st.session_state = MagicMock()
        mock_st.session_state.language = "en"
        mock_st.button.return_value = False
        
        long_content = "a" * 500
        mock_source = {
            "content": long_content,
            "doc_issue": "vol_1_issue_1",
            "volume": "1",
            "heading": "Test",
            "author_name": "Author"
        }
        
        from streamlit_app import render_sources
        
        render_sources(0, [mock_source])
        
        assert mock_st.button.called
    
    @patch('streamlit_app.st')
    def test_render_sources_dict_format(self, mock_st):
        """Test rendering sources in dictionary format (from hybrid_search)."""
        mock_expander = MagicMock()
        mock_st.expander.return_value.__enter__ = Mock(return_value=mock_expander)
        mock_st.expander.return_value.__exit__ = Mock(return_value=None)
        mock_st.session_state.language = "ta"
        
        mock_source = {
            "content": "Test content",
            "doc_issue": "vol_2_issue_3",
            "volume": "2",
            "heading": "கட்டுரை",
            "author_name": "எழுத்தாளர்"
        }
        
        from streamlit_app import render_sources
        
        render_sources(0, [mock_source])
        
        mock_st.expander.assert_called_once()
        # Check that Tamil translations are used
        assert mock_st.markdown.called
    
    @patch('streamlit_app.st')
    def test_render_sources_missing_metadata(self, mock_st):
        """Test rendering sources with missing metadata fields."""
        mock_expander = MagicMock()
        mock_st.expander.return_value.__enter__ = Mock(return_value=mock_expander)
        mock_st.expander.return_value.__exit__ = Mock(return_value=None)
        mock_st.session_state.language = "en"
        
        mock_source = Mock()
        mock_source.payload = {
            "content": "Content only",
            "metadata": {}
        }
        
        from streamlit_app import render_sources
        
        render_sources(0, [mock_source])
        
        # Should handle missing metadata gracefully
        assert mock_st.markdown.called
    
    @patch('streamlit_app.st')
    def test_render_sources_read_more_toggle(self, mock_st):
        """Test read more/less toggle functionality."""
        mock_expander = MagicMock()
        mock_st.expander.return_value.__enter__ = Mock(return_value=mock_expander)
        mock_st.expander.return_value.__exit__ = Mock(return_value=None)
        
        # Use MagicMock for session_state to support attribute access
        mock_st.session_state = MagicMock()
        mock_st.session_state.language = "en"
        
        # First call: button returns True (toggle)
        mock_st.button.return_value = True
        
        # Mock st.rerun() to raise exception
        mock_st.rerun.side_effect = Exception("Rerun triggered")
        
        long_content = "a" * 500
        mock_source = {
            "content": long_content,
            "doc_issue": "vol_1_issue_1",
            "volume": "1",
            "heading": "Test",
            "author_name": "Author"
        }
        
        from streamlit_app import render_sources
        
        with pytest.raises(Exception, match="Rerun triggered"):
            render_sources(0, [mock_source])


class TestRenderLibraryPage:
    """Test suite for library page rendering."""
    
    @patch('streamlit_app.st')
    @patch('streamlit_app.render_volume_card')
    def test_render_library_page(self, mock_render_card, mock_st):
        """Test library page renders all volumes."""
        mock_st.session_state.language = "en"
        
        def columns_side_effect(spec, **kwargs):
            if isinstance(spec, int):
                num_cols = spec
            elif isinstance(spec, list):
                num_cols = len(spec)
            else:
                num_cols = spec
            
            cols = []
            for _ in range(num_cols):
                col = MagicMock()
                col.__enter__ = Mock(return_value=col)
                col.__exit__ = Mock(return_value=None)
                cols.append(col)
            return cols
        
        mock_st.columns.side_effect = columns_side_effect
        
        from streamlit_app import render_library_page
        
        render_library_page()
        
        assert mock_render_card.call_count == 8


class TestRenderVolumeCard:
    """Test suite for volume card rendering."""

    @patch('streamlit_app.st')
    def test_render_volume_card_with_cover(self, mock_st):
        """Test successful volume card rendering with S3 cover URL."""
        mock_st.session_state.language = "en"

        volume = {"id": 1, "desc": "1947", "cover_url": "https://s3.example.com/Volume1.jpg"}

        from streamlit_app import render_volume_card
        render_volume_card(volume)

        mock_st.markdown.assert_called_once()
        card_html = mock_st.markdown.call_args[0][0]
        assert "s3.example.com" in card_html

    @patch('streamlit_app.st')
    def test_render_volume_card_no_cover(self, mock_st):
        """Test volume card with no cover URL shows placeholder."""
        mock_st.session_state.language = "en"

        volume = {"id": 1, "desc": "1947", "cover_url": None}

        from streamlit_app import render_volume_card
        render_volume_card(volume)

        card_html = mock_st.markdown.call_args[0][0]
        assert "No cover" in card_html


class TestSetPage:
    """Test suite for page navigation helper."""
    
    @patch('streamlit_app.st')
    def test_set_page_single_param(self, mock_st):
        """Test setting single page parameter."""
        mock_st.query_params = MagicMock()
        mock_st.query_params.__iter__ = Mock(return_value=iter([]))
        
        from streamlit_app import set_page
        
        set_page(page="library")
        
        mock_st.query_params.clear.assert_called_once()
        mock_st.query_params.update.assert_called_once()
    
    @patch('streamlit_app.st')
    def test_set_page_multiple_params(self, mock_st):
        """Test setting multiple parameters."""
        mock_st.query_params = MagicMock()
        mock_st.query_params.__iter__ = Mock(return_value=iter([]))
        
        from streamlit_app import set_page
        
        set_page(page="pdf_viewer", volume="2", issue="5")
        
        mock_st.query_params.update.assert_called_once()
    
    @patch('streamlit_app.st')
    def test_set_page_with_none_values(self, mock_st):
        """Test set_page filters out None values."""
        mock_st.query_params = MagicMock()
        mock_st.query_params.__iter__ = Mock(return_value=iter([]))
        
        from streamlit_app import set_page
        
        set_page(page="issues", volume="3", issue=None)
        
        # Should only include non-None values
        call_args = mock_st.query_params.update.call_args[0][0]
        assert "page" in call_args
        assert "volume" in call_args
        assert "issue" not in call_args or call_args.get("issue") is None


class TestRenderIssuesPage:
    """Test suite for issues page rendering."""
    
    @patch('streamlit_app.st')
    @patch('streamlit_app.load_volume_issues')
    @patch('streamlit_app.render_issue_grid')
    def test_render_issues_page_with_issues(self, mock_render_grid, mock_load_issues, mock_st):
        """Test rendering issues page with available issues."""
        mock_st.session_state.language = "en"
        mock_st.button.return_value = False
        mock_load_issues.return_value = [
            {"issue_num": "1", "has_pdf": True, "cover_url": "https://s3.example.com/i1.jpg"},
            {"issue_num": "2", "has_pdf": True, "cover_url": "https://s3.example.com/i2.jpg"}
        ]

        from streamlit_app import render_issues_page

        render_issues_page("1")

        mock_load_issues.assert_called_once()
        mock_render_grid.assert_called_once()

    @patch('streamlit_app.st')
    @patch('streamlit_app.load_volume_issues')
    def test_render_issues_page_no_issues(self, mock_load_issues, mock_st):
        """Test rendering issues page with no issues found."""
        mock_st.session_state.language = "en"
        mock_st.button.return_value = False
        mock_load_issues.return_value = []

        from streamlit_app import render_issues_page

        render_issues_page("1")

        mock_load_issues.assert_called_once()

    @patch('streamlit_app.st')
    @patch('streamlit_app.load_volume_issues')
    @patch('streamlit_app.set_page')
    def test_render_issues_page_back_button(self, mock_set_page, mock_load_issues, mock_st):
        """Test back button functionality on issues page."""
        mock_st.session_state.language = "en"
        mock_st.button.return_value = True  # Back button clicked
        mock_load_issues.return_value = []

        # Mock st.rerun() to raise exception
        mock_st.rerun.side_effect = Exception("Rerun triggered")
        
        from streamlit_app import render_issues_page
        
        with pytest.raises(Exception, match="Rerun triggered"):
            render_issues_page("1")


class TestLoadVolumeIssues:
    """Test suite for loading volume issues."""
    
    @patch('streamlit_app.PDF_LINKS', {
        "vol_1_issue_1": "url1",
        "vol_1_issue_2": "url2"
    })
    def test_load_volume_issues_success(self):
        """Test loading volume issues successfully."""
        test_folder = Path("/test/volume 1 cover images")
        
        with patch.object(Path, 'exists', return_value=True), \
             patch.object(Path, 'glob') as mock_glob:
            
            mock_glob.return_value = [
                Path("/test/volume 1 cover images/issue1.jpg"),
                Path("/test/volume 1 cover images/issue2.jpg")
            ]
            
            from streamlit_app import load_volume_issues
            
            issues = load_volume_issues("1", test_folder)
            
            assert len(issues) == 2
            assert issues[0]["issue_num"] == 1
            assert issues[0]["has_pdf"] is True
    
    def test_load_volume_issues_missing_folder(self):
        """Test loading issues from non-existent folder."""
        test_folder = Path("/nonexistent/folder")
        
        with patch.object(Path, 'exists', return_value=False):
            from streamlit_app import load_volume_issues
            
            issues = load_volume_issues("1", test_folder)
            
            assert issues == []
    
    @patch('streamlit_app.PDF_LINKS', {
        "vol_2_issue_1": "url1"
    })
    def test_load_volume_issues_multiple_extensions(self):
        """Test loading issues with different image extensions."""
        test_folder = Path("/test/volume 2 cover images")
        
        with patch.object(Path, 'exists', return_value=True), \
             patch.object(Path, 'glob') as mock_glob:
            
            # Mock glob to return different results for different patterns
            def glob_side_effect(pattern):
                if pattern == '*.jpg':
                    return [Path("/test/volume 2 cover images/issue1.jpg")]
                elif pattern == '*.png':
                    return [Path("/test/volume 2 cover images/issue2.png")]
                elif pattern == '*.jpeg':
                    return [Path("/test/volume 2 cover images/issue3.jpeg")]
                else:
                    return []
            
            mock_glob.side_effect = glob_side_effect
            
            from streamlit_app import load_volume_issues
            
            issues = load_volume_issues("2", test_folder)
            
            # Only issue 1 should have PDF link
            assert any(issue["has_pdf"] for issue in issues)


class TestRenderIssueGrid:
    """Test suite for issue grid rendering."""
    
    @patch('streamlit_app.st')
    @patch('streamlit_app.render_issue_card')
    def test_render_issue_grid_multiple_rows(self, mock_render_card, mock_st):
        """Test rendering issue grid with multiple rows."""
        def columns_side_effect(spec, **kwargs):
            if isinstance(spec, int):
                num_cols = spec
            elif isinstance(spec, list):
                num_cols = len(spec)
            else:
                num_cols = spec
            
            cols = []
            for _ in range(num_cols):
                col = MagicMock()
                col.__enter__ = Mock(return_value=col)
                col.__exit__ = Mock(return_value=None)
                cols.append(col)
            return cols
        
        mock_st.columns.side_effect = columns_side_effect
        
        issues = [
            {"issue_num": i, "has_pdf": True, "image_path": Path(f"/test/i{i}.jpg")}
            for i in range(1, 9)
        ]
        
        from streamlit_app import render_issue_grid
        
        render_issue_grid(issues, "1")
        
        assert mock_render_card.call_count == 8


class TestRenderIssueCard:
    """Test suite for issue card rendering."""
    
    @patch('streamlit_app.st')
    @patch('streamlit_app.Image')
    def test_render_issue_card_success(self, mock_pil, mock_st):
        """Test successful issue card rendering."""
        mock_img = Mock()
        mock_img.mode = "RGB"
        mock_pil.open.return_value = mock_img
        mock_st.session_state.language = "en"
        
        issue = {
            "issue_num": 1,
            "has_pdf": True,
            "image_path": Path("/test/issue1.jpg")
        }
        
        from streamlit_app import render_issue_card
        
        render_issue_card(issue, "1")
        
        mock_st.markdown.assert_called_once()
    
    @patch('streamlit_app.st')
    @patch('streamlit_app.Image')
    @patch('streamlit_app.logger')
    def test_render_issue_card_with_rgba_image(self, mock_logger, mock_pil, mock_st):
        """Test issue card with RGBA image that needs conversion."""
        mock_img = Mock()
        mock_img.mode = "RGBA"
        converted_img = Mock()
        converted_img.mode = "RGB"
        mock_img.convert.return_value = converted_img
        mock_pil.open.return_value = mock_img
        mock_st.session_state.language = "ta"
        
        issue = {
            "issue_num": 2,
            "has_pdf": True,
            "image_path": Path("/test/issue2.png")
        }
        
        from streamlit_app import render_issue_card
        
        render_issue_card(issue, "1")
        
        mock_img.convert.assert_called_with("RGB")
    
    @patch('streamlit_app.st')
    @patch('streamlit_app.Image')
    @patch('streamlit_app.logger')
    def test_render_issue_card_exception_handling(self, mock_logger, mock_pil, mock_st):
        """Test issue card handles image loading exceptions."""
        mock_pil.open.side_effect = Exception("Image error")
        mock_st.session_state.language = "en"
        
        issue = {
            "issue_num": 1,
            "has_pdf": True,
            "image_path": Path("/test/bad.jpg")
        }
        
        from streamlit_app import render_issue_card
        
        render_issue_card(issue, "1")
        
        mock_logger.error.assert_called()


class TestRenderPDFViewerPage:
    """Test suite for PDF viewer page rendering."""
    
    @patch('streamlit_app.st')
    @patch('streamlit_app.PDF_LINKS', {"vol_1_issue_1": "https://drive.google.com/file/d/ABC123/view"})
    def test_render_pdf_viewer_success(self, mock_st):
        """Test successful PDF viewer rendering."""
        mock_st.session_state.language = "en"
        mock_st.button.return_value = False
        
        from streamlit_app import render_pdf_viewer_page
        
        render_pdf_viewer_page("1", "1")
        
        markdown_calls = [call[0][0] for call in mock_st.markdown.call_args_list]
        assert any("iframe" in str(call) for call in markdown_calls)
    
    @patch('streamlit_app.st')
    @patch('streamlit_app.PDF_LINKS', {})
    def test_render_pdf_viewer_missing_pdf(self, mock_st):
        """Test PDF viewer with missing PDF."""
        mock_st.session_state.language = "en"
        mock_st.button.return_value = False
        
        from streamlit_app import render_pdf_viewer_page
        
        render_pdf_viewer_page("1", "1")
        
        # Should return early without rendering iframe
    
    @patch('streamlit_app.st')
    @patch('streamlit_app.set_page')
    @patch('streamlit_app.PDF_LINKS', {"vol_2_issue_3": "https://drive.google.com/file/d/XYZ789/view"})
    def test_render_pdf_viewer_back_button(self, mock_set_page, mock_st):
        """Test back button on PDF viewer page."""
        mock_st.session_state.language = "en"
        mock_st.button.return_value = True  # Back button clicked
        
        # Mock st.rerun() to raise exception
        mock_st.rerun.side_effect = Exception("Rerun triggered")
        
        from streamlit_app import render_pdf_viewer_page
        
        with pytest.raises(Exception, match="Rerun triggered"):
            render_pdf_viewer_page("2", "3")
    
    @patch('streamlit_app.st')
    @patch('streamlit_app.extract_file_id')
    @patch('streamlit_app.PDF_LINKS', {"vol_1_issue_1": "invalid_url"})
    @patch('streamlit_app.logger')
    def test_render_pdf_viewer_invalid_url(self, mock_logger, mock_extract, mock_st):
        """Test PDF viewer with invalid URL format."""
        mock_st.session_state.language = "en"
        mock_st.button.return_value = False
        mock_extract.return_value = None  # Invalid URL
        
        from streamlit_app import render_pdf_viewer_page
        
        render_pdf_viewer_page("1", "1")
        
        # Should log error and return early


class TestExtractFileId:
    """Test suite for Google Drive file ID extraction."""
    
    def test_extract_file_id_with_id_param(self):
        """Test extracting file ID from URL with id= parameter."""
        from streamlit_app import extract_file_id
        
        url = "https://drive.google.com/file?id=ABC123&extra=param"
        result = extract_file_id(url)
        
        assert result == "ABC123"
    
    def test_extract_file_id_with_d_format(self):
        """Test extracting file ID from URL with /d/ format."""
        from streamlit_app import extract_file_id
        
        url = "https://drive.google.com/file/d/XYZ789/view"
        result = extract_file_id(url)
        
        assert result == "XYZ789"
    
    def test_extract_file_id_none_url(self):
        """Test extracting file ID from None URL."""
        from streamlit_app import extract_file_id
        
        result = extract_file_id(None)
        
        assert result is None
    
    def test_extract_file_id_invalid_format(self):
        """Test extracting file ID from invalid URL format."""
        from streamlit_app import extract_file_id
        
        url = "https://example.com/invalid"
        result = extract_file_id(url)
        
        assert result is None
    
    def test_extract_file_id_malformed_url(self):
        """Test extracting file ID from malformed URL."""
        from streamlit_app import extract_file_id
        
        url = "not a valid url at all"
        result = extract_file_id(url)
        
        assert result is None


class TestLoadImage:
    """Test suite for image loading utility."""
    
    @patch('streamlit_app.st')
    @patch('streamlit_app._volume_cover_bytes')
    def test_load_image_success(self, mock_cover_bytes, mock_st):
        """Test loading image via API proxy using st.image."""
        mock_cover_bytes.return_value = b'\xff\xd8\xff\xe0fake-jpeg'

        from streamlit_app import load_image

        load_image("Volume1")

        mock_st.image.assert_called_once()
        assert mock_st.image.call_args[0][0] == b'\xff\xd8\xff\xe0fake-jpeg'

    @patch('streamlit_app.st')
    @patch('streamlit_app.logger')
    def test_load_image_not_found(self, mock_logger, mock_st):
        """Test loading image with invalid name."""
        from streamlit_app import load_image

        load_image("nonexistent")

        mock_logger.warning.assert_called()


class TestRenderAboutPage:
    """Test suite for about page rendering."""
    
    @patch('streamlit_app.st')
    @patch('streamlit_app.load_image')
    def test_render_about_page(self, mock_load_image, mock_st):
        """Test about page rendering."""
        def columns_side_effect(spec, **kwargs):
            if isinstance(spec, int):
                num_cols = spec
            elif isinstance(spec, list):
                num_cols = len(spec)
            else:
                num_cols = spec
            
            cols = []
            for _ in range(num_cols):
                col = MagicMock()
                col.__enter__ = Mock(return_value=col)
                col.__exit__ = Mock(return_value=None)
                cols.append(col)
            return cols
        
        mock_st.columns.side_effect = columns_side_effect
        
        from streamlit_app import render_about_page
        
        render_about_page()
        
        assert mock_st.markdown.call_count > 5
        assert mock_load_image.call_count == 5


class TestTranslationFunction:
    """Test suite for translation helper function."""
    
    @patch('streamlit_app.st')
    def test_translation_tamil(self, mock_st):
        """Test translation function with Tamil language."""
        mock_st.session_state.language = "ta"
        
        from streamlit_app import t
        
        result = t("app_title")
        assert result == "பொன்னி களஞ்சியம்"
    
    @patch('streamlit_app.st')
    def test_translation_english(self, mock_st):
        """Test translation function with English language."""
        mock_st.session_state.language = "en"
        
        from streamlit_app import t
        
        result = t("app_title")
        assert result == "Ponni Archive"
    
    @patch('streamlit_app.st')
    def test_translation_missing_key(self, mock_st):
        """Test translation function with missing key."""
        mock_st.session_state.language = "en"
        
        from streamlit_app import t
        
        result = t("nonexistent_key")
        assert result == "nonexistent_key"  # Returns key itself


class TestInitializeSessionState:
    """Test suite for session state initialization."""
    
    @patch('streamlit_app.st')
    def test_initialize_session_state_first_run(self, mock_st):
        """Test initializing session state on first run."""
        # Use MagicMock to support both dict-style and attribute-style access
        session_state = {}
        
        class SessionStateMock:
            def __contains__(self, key):
                return key in session_state
            def __getitem__(self, key):
                return session_state[key]
            def __setitem__(self, key, value):
                session_state[key] = value
            def __setattr__(self, key, value):
                session_state[key] = value
            def __getattr__(self, key):
                if key in session_state:
                    return session_state[key]
                # Return default value for missing attributes to pass "in" check
                raise AttributeError(f"'{type(self).__name__}' object has no attribute '{key}'")
            def get(self, key, default=None):
                return session_state.get(key, default)
        
        mock_st.session_state = SessionStateMock()
        
        from streamlit_app import initialize_session_state
        
        initialize_session_state()
        
        assert session_state.get("language") == "ta"
        assert isinstance(session_state.get("messages"), list)
    
    @patch('streamlit_app.st')
    def test_initialize_session_state_already_initialized(self, mock_st):
        """Test initializing session state when already initialized."""
        session_state = {"language": "en", "messages": [{"test": "data"}]}
        
        class SessionStateMock:
            def __contains__(self, key):
                return key in session_state
            def __getitem__(self, key):
                return session_state[key]
        
        mock_st.session_state = SessionStateMock()
        
        from streamlit_app import initialize_session_state
        
        initialize_session_state()
        
        # Should not overwrite existing values
        assert session_state["language"] == "en"
        assert len(session_state["messages"]) == 1


class TestGetAppStyles:
    """Test suite for app styles function."""
    
    def test_get_app_styles_returns_css(self):
        """Test that get_app_styles returns CSS string."""
        from streamlit_app import get_app_styles
        
        result = get_app_styles()
        
        assert isinstance(result, str)
        assert "<style>" in result
        assert "</style>" in result
        assert "nav-container" in result
        assert "hero-title" in result


class TestMainFunction:
    """Test suite for main application function."""
    
    @patch('streamlit_app.st')
    @patch('streamlit_app.configure_page')
    @patch('streamlit_app.initialize_session_state')
    @patch('streamlit_app.handle_query_parameters')
    @patch('streamlit_app.get_app_styles')
    @patch('streamlit_app.render_navigation_bar')
    @patch('streamlit_app.render_home_page')
    def test_main_home_page(self, mock_render_home, mock_render_nav, mock_get_styles,
                           mock_handle_params, mock_init_state, mock_config_page, mock_st):
        """Test main function routing to home page."""
        mock_handle_params.return_value = ("home", None, None)
        mock_get_styles.return_value = "<style></style>"
        
        from streamlit_app import main
        
        main()
        
        mock_config_page.assert_called_once()
        mock_init_state.assert_called_once()
        mock_render_home.assert_called_once()
    
    @patch('streamlit_app.st')
    @patch('streamlit_app.configure_page')
    @patch('streamlit_app.initialize_session_state')
    @patch('streamlit_app.handle_query_parameters')
    @patch('streamlit_app.get_app_styles')
    @patch('streamlit_app.render_navigation_bar')
    @patch('streamlit_app.render_library_page')
    def test_main_library_page(self, mock_render_lib, mock_render_nav, mock_get_styles,
                               mock_handle_params, mock_init_state, mock_config_page, mock_st):
        """Test main function routing to library page."""
        mock_handle_params.return_value = ("library", None, None)
        mock_get_styles.return_value = "<style></style>"
        
        from streamlit_app import main
        
        main()
        
        mock_render_lib.assert_called_once()
    
    @patch('streamlit_app.st')
    @patch('streamlit_app.configure_page')
    @patch('streamlit_app.initialize_session_state')
    @patch('streamlit_app.handle_query_parameters')
    @patch('streamlit_app.get_app_styles')
    @patch('streamlit_app.render_navigation_bar')
    @patch('streamlit_app.render_pdf_viewer_page')
    def test_main_pdf_viewer_page(self, mock_render_pdf, mock_render_nav, mock_get_styles,
                                  mock_handle_params, mock_init_state, mock_config_page, mock_st):
        """Test main function routing to PDF viewer page."""
        mock_handle_params.return_value = ("pdf_viewer", "2", "5")
        mock_get_styles.return_value = "<style></style>"
        
        from streamlit_app import main
        
        main()
        
        mock_render_pdf.assert_called_once_with("2", "5")
    
    @patch('streamlit_app.st')
    @patch('streamlit_app.configure_page')
    @patch('streamlit_app.initialize_session_state')
    @patch('streamlit_app.handle_query_parameters')
    @patch('streamlit_app.get_app_styles')
    @patch('streamlit_app.render_navigation_bar')
    @patch('streamlit_app.render_issues_page')
    def test_main_issues_page(self, mock_render_issues, mock_render_nav, mock_get_styles,
                             mock_handle_params, mock_init_state, mock_config_page, mock_st):
        """Test main function routing to issues page."""
        mock_handle_params.return_value = ("issues", "3", None)
        mock_get_styles.return_value = "<style></style>"
        
        from streamlit_app import main
        
        main()
        
        mock_render_issues.assert_called_once_with("3")
    
    @patch('streamlit_app.st')
    @patch('streamlit_app.configure_page')
    @patch('streamlit_app.initialize_session_state')
    @patch('streamlit_app.handle_query_parameters')
    @patch('streamlit_app.get_app_styles')
    @patch('streamlit_app.render_navigation_bar')
    @patch('streamlit_app.render_about_page')
    def test_main_about_page(self, mock_render_about, mock_render_nav, mock_get_styles,
                            mock_handle_params, mock_init_state, mock_config_page, mock_st):
        """Test main function routing to about page."""
        mock_handle_params.return_value = ("about", None, None)
        mock_get_styles.return_value = "<style></style>"
        
        from streamlit_app import main
        
        main()
        
        mock_render_about.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])