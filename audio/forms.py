from django import forms


class UploadForm(forms.Form):
    audio = forms.FileField(help_text='Upload a short WAV file (mono or stereo).')
    apply_base_denoise = forms.BooleanField(initial=True, required=False, label='Soft denoise')
    suppress_triggers = forms.BooleanField(initial=True, required=False, label='Suppress trigger sounds')
    trigger_sensitivity = forms.FloatField(
        initial=0.15,  # было 0.25
        min_value=0.0, max_value=1.0, required=False,
        help_text='Lower = more aggressive suppression.'
    )

    def clean_audio(self):
        f = self.cleaned_data['audio']
        if not f.name.lower().endswith('.wav'):
            raise forms.ValidationError('Пока принимаем только WAV (.wav). MP3 требует FFmpeg — добавим позже.')
        return f