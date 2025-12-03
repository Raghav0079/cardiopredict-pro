name: 🫀 CardioPredict Pro - Pull Request
description: Contribute improvements to CardioPredict Pro
title: "[TYPE] Brief description of changes"
labels: ["needs-review"]

body:
  - type: markdown
    attributes:
      value: |
        # Contributing to CardioPredict Pro 🏥
        
        Thank you for contributing to this medical AI education project! 
        
        **⚠️ Medical Safety Notice**: All medical-related changes must be thoroughly validated and reviewed by healthcare professionals.

  - type: dropdown
    id: pr_type
    attributes:
      label: Type of Change
      description: What type of change does this PR introduce?
      options:
        - 🐛 Bug Fix - Non-breaking change that fixes an issue
        - ✨ New Feature - Non-breaking change that adds functionality
        - 🔧 Enhancement - Improvement to existing functionality
        - 🏥 Medical Update - Medical accuracy or clinical improvements
        - 📚 Documentation - Documentation updates or improvements
        - 🎨 UI/UX - User interface or experience improvements
        - ⚡ Performance - Performance improvements or optimizations
        - 🔒 Security - Security improvements or fixes
        - 🧪 Testing - Adding or improving tests
        - 🔨 Refactoring - Code refactoring without functionality changes
        - 📦 Dependencies - Dependency updates or changes
        - 🚀 Deployment - Deployment or infrastructure improvements
        - 💥 Breaking Change - Change that would break existing functionality
    validations:
      required: true

  - type: checkboxes
    id: medical_safety
    attributes:
      label: Medical Safety Checklist
      description: For any changes affecting medical functionality
      options:
        - label: Changes maintain educational-only purpose (no clinical use)
        - label: Medical disclaimers are updated if needed
        - label: No real patient data is included in examples or tests
        - label: Medical accuracy has been validated against literature
        - label: Changes don't encourage clinical usage
        - label: N/A - This PR doesn't affect medical functionality

  - type: textarea
    id: description
    attributes:
      label: Description of Changes
      description: Provide a detailed description of your changes
      placeholder: |
        ## Summary
        Brief explanation of what this PR does and why.
        
        ## Changes Made
        - Detailed list of changes
        - Include both functional and technical changes
        - Mention any new dependencies or requirements
        
        ## Medical Considerations (if applicable)
        - How do these changes affect medical accuracy?
        - What clinical guidelines were considered?
        - How do changes impact educational use?
        
        ## Breaking Changes (if applicable)
        - What existing functionality is affected?
        - What migration steps are needed?
        - How to update existing usage?
    validations:
      required: true

  - type: textarea
    id: motivation
    attributes:
      label: Motivation and Context
      description: Why is this change required? What problem does it solve?
      placeholder: |
        ## Problem Statement
        What specific problem does this PR solve?
        
        ## Use Case
        What scenarios or user needs does this address?
        
        ## Background
        Any relevant background information or context.
        
        ## Related Issues
        - Fixes #issue_number
        - Addresses #issue_number
        - Related to #issue_number

  - type: textarea
    id: testing
    attributes:
      label: Testing
      description: How have you tested these changes?
      placeholder: |
        ## Testing Performed
        - [ ] Unit tests pass
        - [ ] Integration tests pass
        - [ ] Manual testing completed
        - [ ] Medical scenarios validated
        - [ ] Cross-platform testing (if applicable)
        - [ ] Performance testing (if applicable)
        
        ## Test Scenarios
        Describe specific test scenarios, especially medical use cases:
        
        1. Test scenario 1
        2. Test scenario 2
        3. Medical validation scenarios
        
        ## Test Results
        Summarize testing results and any issues found.

  - type: textarea
    id: medical_validation
    attributes:
      label: Medical Validation
      description: For medical-related changes, describe validation process
      placeholder: |
        ## Medical Review (if applicable)
        - [ ] Reviewed medical literature and guidelines
        - [ ] Validated against established clinical protocols
        - [ ] Checked for medical accuracy and appropriateness
        - [ ] Ensured educational-only purpose is maintained
        - [ ] Medical disclaimers updated if needed
        
        ## Clinical Guidelines Referenced
        List any medical guidelines, literature, or expert consultations:
        - Reference 1
        - Reference 2
        
        ## Medical Expert Review
        - [ ] Reviewed by healthcare professional
        - [ ] Validated by medical domain expert
        - [ ] N/A - No medical validation needed
        
        Reviewer name and credentials (optional): ________________

  - type: dropdown
    id: breaking_changes
    attributes:
      label: Breaking Changes
      description: Does this PR introduce breaking changes?
      options:
        - No breaking changes
        - Minor breaking changes (with backward compatibility)
        - Major breaking changes (requires migration)
        - Unsure - needs review
    validations:
      required: true

  - type: textarea
    id: migration
    attributes:
      label: Migration Guide
      description: If this introduces breaking changes, how should users migrate?
      placeholder: |
        ## Migration Steps (if applicable)
        
        ### For Users
        1. Step-by-step migration instructions
        2. What needs to be updated in user workflows
        3. Any new requirements or dependencies
        
        ### For Developers
        1. Code changes required
        2. Configuration updates needed
        3. Database migrations (if applicable)
        
        ### Compatibility
        - Backward compatibility considerations
        - Deprecation timeline (if applicable)
        - Support for legacy functionality

  - type: textarea
    id: deployment
    attributes:
      label: Deployment Considerations
      description: Any special deployment or infrastructure considerations?
      placeholder: |
        ## Deployment Notes
        - [ ] No special deployment requirements
        - [ ] Requires environment variable updates
        - [ ] Requires database migrations
        - [ ] Requires new dependencies installation
        - [ ] Affects Hugging Face Spaces deployment
        - [ ] Requires configuration changes
        
        ## Environment Variables
        List any new or changed environment variables:
        - NEW_VAR: Description and example value
        - CHANGED_VAR: What changed and migration steps
        
        ## Dependencies
        List any new dependencies or version changes:
        - package_name==version: Why this dependency is needed
        
        ## Performance Impact
        - Expected impact on prediction speed
        - Memory or storage requirements changes
        - Scalability considerations

  - type: textarea
    id: screenshots
    attributes:
      label: Screenshots/Demos
      description: Visual evidence of your changes (if applicable)
      placeholder: |
        ## Before/After Screenshots
        If your changes affect the UI, please provide screenshots showing:
        
        ### Before
        [Attach screenshot of current state]
        
        ### After  
        [Attach screenshot of new state]
        
        ## Demo Video/GIF
        If applicable, provide a demo showing the new functionality.
        
        ⚠️ **Important**: Do not include screenshots with real patient data!
        Use synthetic/test data only.

  - type: checkboxes
    id: checklist
    attributes:
      label: Contribution Checklist
      description: Please check all applicable items
      options:
        - label: My code follows the project's coding standards
        - label: I have performed a self-review of my code
        - label: I have commented my code, particularly hard-to-understand areas
        - label: I have made corresponding changes to documentation
        - label: My changes generate no new warnings or errors
        - label: I have added tests that prove my fix is effective or feature works
        - label: New and existing unit tests pass locally with my changes
        - label: Any dependent changes have been merged and published

  - type: checkboxes
    id: code_quality
    attributes:
      label: Code Quality Checklist
      description: Code quality and best practices
      options:
        - label: Code is well-documented with clear docstrings
        - label: Functions have appropriate type hints where applicable
        - label: Error handling is implemented appropriately
        - label: Security considerations have been addressed
        - label: Performance implications have been considered
        - label: Code is DRY (Don't Repeat Yourself) and follows SOLID principles
        - label: Medical parameters are validated appropriately
        - label: User input is properly sanitized and validated

  - type: checkboxes
    id: documentation
    attributes:
      label: Documentation Checklist
      description: Documentation and user guidance
      options:
        - label: README.md updated if functionality changed
        - label: API documentation updated (if applicable)
        - label: Medical disclaimers updated if needed
        - label: User guide updated for new features
        - label: Code comments explain medical logic where applicable
        - label: Examples updated to reflect new functionality
        - label: Deployment documentation updated if needed

  - type: textarea
    id: future_work
    attributes:
      label: Future Work
      description: Any follow-up work or improvements planned?
      placeholder: |
        ## Planned Follow-ups
        - Future enhancements planned for this feature
        - Known limitations that could be addressed later
        - Additional testing or validation needed
        - Integration opportunities with other features
        
        ## Technical Debt
        - Any technical debt introduced that should be addressed
        - Refactoring opportunities identified
        - Performance optimizations planned

  - type: dropdown
    id: review_needed
    attributes:
      label: Special Review Requirements
      description: What type of special review does this PR need?
      options:
        - Standard review - General code review
        - Medical expert review - Requires healthcare professional review
        - Security review - Security implications need review
        - Performance review - Performance impact needs assessment
        - Architecture review - Significant architectural changes
        - UI/UX review - User experience design review
        - Documentation review - Major documentation changes
        - No special review needed

  - type: textarea
    id: reviewers
    attributes:
      label: Suggested Reviewers
      description: Who should review this PR?
      placeholder: |
        ## Suggested Reviewers
        - @username1 - Reason for suggestion
        - @username2 - Specific expertise needed
        
        ## Medical Review
        If medical validation is needed:
        - Healthcare professional contact: ________________
        - Medical specialty relevance: ________________
        - Clinical validation requirements: ________________

  - type: markdown
    attributes:
      value: |
        ---
        
        ## 🏥 Medical AI Contribution Guidelines
        
        **Remember:**
        - This tool is for educational and research purposes only
        - All medical changes require validation against established literature
        - User safety and appropriate disclaimers are paramount
        - Contributions should enhance medical education value
        
        **Review Process:**
        1. **Technical Review**: Code quality, functionality, performance
        2. **Medical Review**: Clinical accuracy, educational value, safety
        3. **Security Review**: Data protection, input validation, compliance
        4. **Final Approval**: Maintainer approval and merge
        
        **Thank you for contributing to advancing medical AI education!** 🚀
        
        Your contributions help make cardiovascular risk assessment education more accessible and effective.