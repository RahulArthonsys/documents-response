from django.db import models
from django.conf import settings
from application.custom_models import DateTimeModel


class SubscriptionPlan(DateTimeModel):
    DURATION_CHOICES = [
        (7,   '7 Days'),
        (30,  '1 Month'),
        (90,  '3 Months'),
        (180, '6 Months'),
        (365, '1 Year'),
    ]

    name = models.CharField('Plan Name', max_length=100)
    description = models.TextField('Description', blank=True)
    price = models.DecimalField('Price ($)', max_digits=10, decimal_places=2)
    duration_days = models.PositiveIntegerField('Duration', choices=DURATION_CHOICES, default=30)
    is_active = models.BooleanField('Active', default=True)

    class Meta:
        ordering = ('price',)

    def __str__(self):
        return f"{self.name} – ${self.price}"


class UserSubscription(DateTimeModel):
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='subscriptions'
    )
    plan = models.ForeignKey(
        SubscriptionPlan,
        on_delete=models.CASCADE,
        related_name='subscriptions'
    )
    start_date = models.DateField('Start Date')
    end_date = models.DateField('End Date')
    is_active = models.BooleanField('Active', default=True)

    class Meta:
        ordering = ('-start_date',)

    def __str__(self):
        return f"{self.user} — {self.plan.name}"
